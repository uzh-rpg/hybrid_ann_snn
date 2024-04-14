"""Root package info."""

__version__ = "0.5"
__author__ = "Gianluca Scarpellini"
__author_email__ = "me@scarpellini.dev"
__license__ = "GPLv3"
__copyright__ = 'Copyright (c) 2018-2020, %s.' % __author__

try:
    # This variable is injected in the __builtins__ by the build
    # process.
    __EXPERIMENTING_SETUP__  # type: ignore
except NameError:
    __EXPERIMENTING_SETUP__ = False

if __EXPERIMENTING_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(
        f"Partial import of `{__name__}` during the build process.\n"
    )  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from experimenting import agents, dataset, utils
