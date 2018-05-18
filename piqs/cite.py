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
    citation = ["""@misc{1805.05129,""",
               """Author = {Nathan Shammah and Shahnawaz Ahmed and Neill Lambert
               and Simone De Liberato and Franco Nori},""",
               """Title = {Open quantum systems with local and collective incoherent
               processes: Efficient numerical simulation using permutational invariance},""",
               """Year = {2018},""",
               """Eprint = {arXiv:1805.05129},""",
               """}"""]

    if verbose:
        print("\n".join(citation))

    if not path:
        path = os.getcwd()

    filename = "piqs.bib"
    with open(os.path.join(path, filename), 'w') as f:
        f.write("\n".join(citation))


if __name__ == "__main__":
    cite()
