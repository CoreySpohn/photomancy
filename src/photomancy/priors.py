"""Unified prior layer: distributions over the fitted-leaf vector ``z``.

photomancy-owned and tfp-free. One ``AbstractPrior`` interface serves every backend:
``log_prob`` for BlackJAX / Laplace (folded into the target logdensity), ``sample`` for
SMC's initial particles, and ``forward`` / ``inverse`` / ``log_prob`` for jaxns (via the
``_JaxnsPrior`` adapter in ``backends/nested.py``). Priors are over ``z`` (the raveled
fitted leaves) -- the same space ``build_scene_logdensity`` uses.

Each distribution we need has a closed-form inverse-CDF, so owning them is a few lines
each and keeps the core free of a heavy distribution dependency.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import erf, erfinv, logsumexp

_SQRT2 = 1.4142135623730951  # plain float: no import-time array, no precision lock

_MIXTURE_NO_CUBE = (
    "MixturePrior has no closed-form unit-cube transform; use it with a gradient / "
    "sample backend (Laplace / NUTS / SMC), not jaxns."
)


def _std_normal_quantile(u):
    """Inverse standard-normal CDF on ``u`` in (0, 1)."""
    return _SQRT2 * erfinv(2.0 * u - 1.0)


def _std_normal_cdf(x):
    """Standard-normal CDF."""
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _mvn_logpdf(z, mean, cholesky):
    """Multivariate-Normal log-density at ``z`` for ``mean`` + lower-Cholesky ``L``."""
    d = mean.shape[0]
    w = solve_triangular(cholesky, z - mean, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    return -0.5 * (jnp.sum(w**2) + d * jnp.log(2.0 * jnp.pi) + logdet)


class AbstractPrior(eqx.Module):
    """A prior over the fitted-leaf vector ``z`` of dimension ``ndim``.

    ``forward`` maps a unit-cube draw ``U`` in ``[0, 1]^ndim`` to ``z`` (the inverse-CDF
    transform jaxns needs); ``inverse`` is its inverse; ``log_prob`` is the log-density
    (for BlackJAX / Laplace). ``sample`` defaults to inverse-CDF sampling.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of fitted dimensions (the length of ``z``)."""

    @abstractmethod
    def forward(self, U):
        """Map a unit-cube draw ``U`` in ``[0, 1]^ndim`` to a parameter draw ``z``."""

    @abstractmethod
    def inverse(self, z):
        """Map ``z`` back to the unit cube ``[0, 1]^ndim``."""

    @abstractmethod
    def log_prob(self, z):
        """Prior log-density at ``z``."""

    def sample(self, key, n):
        """Draw ``n`` samples, shape ``(n, ndim)``, by transforming uniform draws."""
        U = jax.random.uniform(key, (n, self.ndim))
        return jax.vmap(self.forward)(U)


class Uniform(AbstractPrior):
    """Independent Uniform on each component; ``low`` / ``high`` are ``(ndim,)``."""

    low: jnp.ndarray
    high: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.low.shape[0]

    def forward(self, U):
        """Affine map of the unit cube onto ``[low, high]``."""
        return self.low + (self.high - self.low) * U

    def inverse(self, z):
        """Map ``z`` back to the unit cube."""
        return (z - self.low) / (self.high - self.low)

    def log_prob(self, z):
        """Constant ``-sum(log(high-low))`` inside the box, ``-inf`` outside."""
        in_box = jnp.all((z >= self.low) & (z <= self.high))
        lp = -jnp.sum(jnp.log(self.high - self.low))
        return jnp.where(in_box, lp, -jnp.inf)


class Normal(AbstractPrior):
    """Independent Normal on each component; ``loc`` / ``scale`` are ``(ndim,)``."""

    loc: jnp.ndarray
    scale: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.loc.shape[0]

    def forward(self, U):
        """Normal inverse-CDF of the unit cube."""
        return self.loc + self.scale * _std_normal_quantile(U)

    def inverse(self, z):
        """Normal CDF of ``z``."""
        return _std_normal_cdf((z - self.loc) / self.scale)

    def log_prob(self, z):
        """Gaussian log-density summed over components."""
        return jnp.sum(
            -0.5 * ((z - self.loc) / self.scale) ** 2
            - jnp.log(self.scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )


class LogNormal(AbstractPrior):
    """Independent LogNormal; ``loc`` / ``scale`` are the underlying Normal's params."""

    loc: jnp.ndarray
    scale: jnp.ndarray

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.loc.shape[0]

    def _normal(self):
        """The underlying Normal in log-space."""
        return Normal(loc=self.loc, scale=self.scale)

    def forward(self, U):
        """Exponentiate the underlying Normal's inverse-CDF."""
        return jnp.exp(self._normal().forward(U))

    def inverse(self, z):
        """Underlying Normal CDF of ``log z``."""
        return self._normal().inverse(jnp.log(z))

    def log_prob(self, z):
        """LogNormal log-density (Normal in log-space plus the ``-log z`` Jacobian)."""
        return self._normal().log_prob(jnp.log(z)) - jnp.sum(jnp.log(z))


class IndependentPrior(AbstractPrior):
    """Product of per-leaf priors, concatenated in z-order."""

    priors: tuple

    @property
    def ndim(self):
        """Total dimensions across the component priors."""
        return sum(p.ndim for p in self.priors)

    def _blocks(self, v):
        """Split a ``(ndim,)`` vector into per-prior slices."""
        out, i = [], 0
        for p in self.priors:
            out.append(v[i : i + p.ndim])
            i += p.ndim
        return out

    def forward(self, U):
        """Per-component ``forward``, concatenated."""
        return jnp.concatenate(
            [p.forward(b) for p, b in zip(self.priors, self._blocks(U), strict=True)]
        )

    def inverse(self, z):
        """Per-component ``inverse``, concatenated."""
        return jnp.concatenate(
            [p.inverse(b) for p, b in zip(self.priors, self._blocks(z), strict=True)]
        )

    def log_prob(self, z):
        """Sum of the component log-densities."""
        return sum(
            p.log_prob(b)
            for p, b in zip(self.priors, self._blocks(z), strict=True)
        )


class JointPrior(AbstractPrior):
    """A multivariate-Normal prior over ``z`` with ``cov = cholesky @ cholesky.T``.

    What ``to_prior()`` produces for a Gaussian (Laplace) posterior, so a fit's
    posterior carries its full covariance into the next fit's prior.
    """

    mean: jnp.ndarray
    cholesky: jnp.ndarray  # lower-triangular L

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.mean.shape[0]

    def forward(self, U):
        """``mean + L @ Phi^-1(U)`` -- affine of standard normals."""
        return self.mean + self.cholesky @ _std_normal_quantile(U)

    def inverse(self, z):
        """Standard-normal CDF of the whitened ``z``."""
        w = solve_triangular(self.cholesky, z - self.mean, lower=True)
        return _std_normal_cdf(w)

    def log_prob(self, z):
        """Multivariate-Normal log-density via the triangular solve."""
        return _mvn_logpdf(z, self.mean, self.cholesky)


class MixturePrior(AbstractPrior):
    """A Gaussian-mixture prior over ``z`` -- weighted modes, each an MVN.

    What ``MixturePosterior.to_prior()`` (and ``SamplePosterior.to_prior`` via
    ``cluster_to_mixture``) produce: it preserves multimodality across sequential epochs
    where a single Gaussian would collapse the modes. ``log_prob`` and ``sample`` are
    exact (serving the Laplace / NUTS / SMC family); ``forward`` / ``inverse`` -- the
    jaxns unit-cube transform -- are not implemented, as a multivariate mixture has no
    closed-form inverse-CDF (use a gradient / sample backend with a mixture prior).
    """

    means: jnp.ndarray  # (K, d)
    choleskys: jnp.ndarray  # (K, d, d), each lower-triangular
    log_weights: jnp.ndarray  # (K,), unnormalized log mixture weights

    @property
    def ndim(self):
        """Number of dimensions ``d``."""
        return self.means.shape[1]

    def log_prob(self, z):
        """Mixture log-density: ``logsumexp_k(log w_k + N(z | mode_k))``."""
        logw = self.log_weights - logsumexp(self.log_weights)
        comp = jax.vmap(lambda m, L: _mvn_logpdf(z, m, L))(self.means, self.choleskys)
        return logsumexp(logw + comp)

    def sample(self, key, n):
        """Draw ``n`` samples: pick a mode by weight, then sample its Gaussian."""
        k_comp, k_draw = jax.random.split(key)
        comp = jax.random.categorical(k_comp, self.log_weights, shape=(n,))
        keys = jax.random.split(k_draw, n)

        def draw(k, idx):
            std = jax.random.normal(k, (self.ndim,))
            return self.means[idx] + self.choleskys[idx] @ std

        return jax.vmap(draw)(keys, comp)

    def forward(self, U):
        """Not implemented -- a multivariate mixture has no closed-form inverse-CDF."""
        raise NotImplementedError(_MIXTURE_NO_CUBE)

    def inverse(self, z):
        """Not implemented -- a multivariate mixture has no closed-form inverse-CDF."""
        raise NotImplementedError(_MIXTURE_NO_CUBE)
