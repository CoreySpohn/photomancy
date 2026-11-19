"""Score and predict in one call for the tests that compare both."""


def both(term, model, **keywords):
    """Return ``(log_likelihood, prediction)`` of a term at ``model``."""
    return term(model, **keywords), term.predict(model)
