# -*- coding: utf-8 -*-

"""
Import aedat file.
"""

from .ImportAedatDataVersion1or2 import import_aedat_dataversion1or2
from .ImportAedatHeaders import import_aedat_headers


def import_aedat(args):
    """

    Parameters
    ----------
    args :

    Returns
    -------

    """

    output = {'info': args}

    with open(output['info']['filePathAndName'], 'rb') as output['info']['fileHandle']:
        output['info'] = import_aedat_headers(output['info'])
        return import_aedat_dataversion1or2(output['info'])
