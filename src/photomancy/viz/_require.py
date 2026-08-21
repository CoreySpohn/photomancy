"""Import guard for the optional eyepiece dependency."""


def eyepiece():
    """Import eyepiece, or raise with the install hint.

    Returns:
        The imported ``eyepiece`` module.

    Raises:
        ImportError: If eyepiece is not installed; the message names the
            ``photomancy[viz]`` extra that provides it.
    """
    try:
        import eyepiece
    except ImportError:
        raise ImportError(
            "photomancy.viz requires eyepiece: pip install 'photomancy[viz]'"
        ) from None
    return eyepiece
