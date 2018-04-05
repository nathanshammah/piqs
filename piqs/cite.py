"""
Citation generator for PIQS
"""
import sys
import os


def cite(path=None, verbose=True):
    """
    Citation information and bibtex generator for PIQS

    Parameters
    ----------
    path: str
        The complete directory path to generate the bibtex file.
        If not specified then the citation will be generated in cwd
    """
    citation = []

    if verbose:
        print("\n".join(citation))

    if not path:
        path = os.getcwd()

    filename = "qutip.bib"
    with open(os.path.join(path, filename), 'w') as f:
        f.write("\n".join(citation))


if __name__ == "__main__":
    cite()
