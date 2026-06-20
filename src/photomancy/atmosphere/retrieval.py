"""Build a retrieval logdensity: fit atmosphere leaves to a reflected-light spectrum.

The atmosphere parallel of ``build_disk_logdensity`` -- it delegates to the
shared ``core.build_gaussian_fit`` (a Gaussian-spectrum likelihood over the selected
leaves), so the retrieval target (which leaves + which prior) and the forward strategy
(which skyscapes model) are the only atmosphere-specific choices the caller makes.
"""

from photomancy.core import build_gaussian_fit


def build_retrieval_logdensity(
    pm, spectrum, *, fit_leaves, noise_sigma, forward, prior=None
):
    """Fit atmosphere leaves to a reflected-light ``spectrum`` (Gaussian noise).

    Args:
        pm: A skyscapes ``ExoJaxPhysicalModel`` (its selected leaves are fit).
        spectrum: Observed contrast spectrum, shape ``(n_wavelengths,)``.
        fit_leaves: ``pm -> list[leaf]`` selecting the retrieval target's leaves (e.g.
            :func:`~photomancy.atmosphere.abundance_fit_leaves`).
        noise_sigma: Per-wavelength Gaussian noise sigma (scalar or broadcastable).
        forward: ``pm -> spectrum`` (from ``contrast_forward``, or a coronagraph / IFS
            forward).
        prior: Optional ``AbstractPrior`` (e.g.
            :func:`~photomancy.atmosphere.default_abundance_prior`) or ``pm -> scalar``
            callable; ``None`` is flat (improper).

    Returns:
        ``(logdensity, z0, unravel)`` from :func:`~photomancy.core.build_gaussian_fit`.
    """
    return build_gaussian_fit(
        pm,
        spectrum,
        fit_leaves=fit_leaves,
        noise_sigma=noise_sigma,
        forward=forward,
        prior=prior,
    )
