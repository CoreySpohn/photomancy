"""Reflected-light forward wrappers for atmosphere retrieval.

photomancy reimplements no atmosphere physics: ``contrast_forward`` wraps a skyscapes
``ExoJaxPhysicalModel``'s own reflected-light render at a fixed observing geometry, so
the fit sees a plain ``pm -> spectrum`` callable. It is forward-strategy agnostic -- the
fixed-T precomputed ``for_retrieval`` and the live-recompute ``from_default_setup``
both expose ``contrast_cube`` -- so swapping the strategy never touches the fit.
"""


def contrast_forward(*, phase, dist_pc, wavelengths_nm, Rp):
    """Build the atmosphere forward: ``pm -> reflected-light contrast spectrum``.

    Reads ``pm.contrast_cube`` at the given observing geometry and returns the single
    planet / single column spectrum of shape ``(n_wavelengths,)``. A coronagraph / IFS
    forward can drop in with the same ``pm -> spectrum`` signature.

    Args:
        phase: Orbital phase angle(s), shape ``(K, 1)`` (skyscapes convention).
        dist_pc: Distance(s), shape ``(K, 1)``.
        wavelengths_nm: Wavelength grid, shape ``(n_wavelengths,)``.
        Rp: Planet radius/radii, shape ``(K,)``.

    Returns:
        ``forward(pm) -> spectrum`` of shape ``(n_wavelengths,)``.
    """

    def forward(pm):
        return pm.contrast_cube(phase, dist_pc, wavelengths_nm, Rp)[:, 0, 0]

    return forward
