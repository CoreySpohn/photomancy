"""Reflected-light forward wrappers for atmosphere retrieval.

photomancy reimplements no atmosphere physics: ``contrast_forward`` wraps a skyscapes
``ExoJaxPhysicalModel``'s own reflected-light render at a fixed observing geometry, so
the fit sees a plain ``pm -> spectrum`` callable. It is forward-strategy agnostic -- the
fixed-T precomputed ``for_retrieval`` and the live-recompute ``from_default_setup``
both expose ``contrast_cube`` -- so swapping the strategy never touches the fit.
"""


def contrast_forward(*, phase_angle_rad, dist_AU, wavelengths_nm, Rp_Rearth):
    """Build the atmosphere forward: ``pm -> reflected-light contrast spectrum``.

    Reads ``pm.contrast_cube`` at the given observing geometry and returns the single
    planet / single column spectrum of shape ``(n_wavelengths,)``. A coronagraph / IFS
    forward can drop in with the same ``pm -> spectrum`` signature.

    Args:
        phase_angle_rad: Orbital phase angles in radians, shape ``(K, 1)``.
        dist_AU: Star-planet distances in AU, shape ``(K, 1)``.
        wavelengths_nm: Wavelength grid in nm, shape ``(n_wavelengths,)``.
        Rp_Rearth: Planet radii in Earth radii, shape ``(K,)``.

    Returns:
        ``forward(pm) -> spectrum`` of shape ``(n_wavelengths,)``.
    """

    def forward(pm):
        return pm.contrast_cube(
            phase_angle_rad=phase_angle_rad,
            star_planet_distance_AU=dist_AU,
            wavelengths_nm=wavelengths_nm,
            Rp_Rearth=Rp_Rearth,
        )[:, 0, 0]

    return forward
