"""Event and jet selections shared by every sample type.

This is the single place where cuts are defined. Both graph production
(``FileListGraphProducer``) and scoring (``score_gae.py``) import from here,
so signal, background and data are always selected identically.

The selections below are intentionally empty placeholders: fill in the
analysis cuts here, and every stage of the pipeline picks them up.

Branches available in the scouting PFNanoAOD skims, for reference:

    event level : ht, ScoutMET_pt, ScoutMET_phi, RTFatJet, MT01FatJetMET,
                  DeltaPhiMinFatJetMET, DeltaEtaJ0J1FatJet, nFatJet,
                  nVetoLeptons, nPV, scouting_trig
    jet level   : FatJet_isGood, FatJet_passId, FatJet_pt, FatJet_eta,
                  FatJet_mass, FatJet_nConstituents

Any branch used below must also be listed in ``CUT_BRANCHES`` so that it is
read from the ROOT file.
"""

import hashlib
import inspect

import numpy as np


# Extra branches the selections below need, beyond those the graph builder
# already loads. Keep in sync with event_cut_mask and jet_cut_mask.
CUT_BRANCHES = []


def event_cut_mask(events):
    """Return a boolean mask over the entries in *events*.

    Applied identically to signal, background and data. *events* is the
    awkward record array read from the ``Events`` tree, containing at least
    the graph-building branches plus ``CUT_BRANCHES``.
    """
    mask = np.ones(len(events), dtype=bool)

    # TODO: add the analysis event selection, for example
    # mask = mask & (np.asarray(events["ht"]) > 0.0)
    # mask = mask & (np.asarray(events["nVetoLeptons"]) == 0)

    return mask


def jet_cut_mask(event, jet_index, config):
    """Return True when the jet at *jet_index* should be turned into a graph.

    Applied identically to signal, background and data. *event* is a single
    entry of the awkward array and *config* is the ``GraphBuildConfig`` in
    use, so ``config.jet_collection`` gives the collection name.
    """
    # TODO: add the analysis jet selection, for example
    # collection = config.jet_collection
    # return bool(event[f"{collection}_isGood"][jet_index])

    return True


def cuts_fingerprint():
    """Stable hash of the selection source, used to invalidate graph caches."""
    source = "".join(
        inspect.getsource(function)
        for function in (event_cut_mask, jet_cut_mask)
    )
    source += repr(sorted(CUT_BRANCHES))
    return hashlib.sha256(source.encode()).hexdigest()
