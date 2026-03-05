from __future__ import annotations

import logging

import awkward as ak
import numpy as np
import uproot
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.processing.features import resolve_features

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_path(
    cfg: DictConfig,
    sample_type: str,
    campaign: str,
    sample_id: str,
) -> str:
    """Format the ROOT file path from the paths config."""
    template = cfg.paths[sample_type]
    return template.format(
        analysis_base=cfg.analysis.analysis_base,
        campaign=campaign,
        sample=sample_id,
    )


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------


def _ntau_from_channel(channel: str) -> str:
    """Map a channel string to its expected tau multiplicity string."""
    if channel == "2":
        return "2"
    if channel in ("1had0lep", "1had1lep", "1"):
        return "1"
    if channel == "0":
        return "0"
    raise ValueError(
        f"Unsupported channel '{channel}'. "
        "Only '0', '1had0lep', '1had1lep', '1', or '2' allowed."
    )


def _apply_channel_cuts(ar: ak.Array, channel: str, ntau: str, region: str) -> ak.Array:
    """Tau multiplicity and lepton-veto cuts."""
    if region != "CR":
        if ntau == "0":
            ar = ar[ar.tau_n == 0]
            ar = ar[ar.mu_n == 1]
        elif ntau == "1":
            ar = ar[ar.tau_n == 1]
        elif ntau == "2":
            ar = ar[ar.tau_n >= 2]

    if channel == "1had0lep":
        ar = ar[ar.LeptonVeto == 1]
    elif channel == "1had1lep":
        ar = ar[ar.LeptonVeto == 0]

    return ar


def _apply_cleaning_cuts(ar: ak.Array) -> ak.Array:
    """Apply event quality and MET trigger cleaning cuts."""
    ar = ar[ar.eventClean != 0]
    ar = ar[ar.isBadTile == 0]
    ar = ar[ar.jet_isBadTight[:, 0] != True]  # noqa: E712
    ar = ar[ar.IsMETTrigPassed != 0]
    return ar


def _apply_rnn_cuts(ar: ak.Array, ntau: str) -> ak.Array:
    """Apply tau RNN medium working point cuts for the given tau multiplicity."""
    if ntau == "1":
        ar = ar[ar.tau_JetRNNMedium[:, 0] != 0]
    elif ntau == "2":
        ar = ar[ar.tau_JetRNNMedium[:, 0] != 0]
        ar = ar[ar.tau_JetRNNMedium[:, 1] != 0]
    return ar


def _apply_truth_cuts(
    ar: ak.Array,
    ntau: str,
    sample_type: str,
    scope: str,
) -> ak.Array:
    """Apply truth-matching cuts for background samples outside the NTuples scope."""
    if scope == "NTuples":
        return ar

    # if sample_type == "background":
    #     if ntau == "1":
    #         ar = ar[ar.tau_isTruthMatchedTau[:, 0] == 1]
    #         ar = ar[ar.tau_truthType[:, 0] == 10]
    #     elif ntau == "2":
    #         ar = ar[ar.tau_isTruthMatchedTau[:, 0] == 1]
    #         ar = ar[ar.tau_truthType[:, 0] == 10]
    #         ar = ar[ar.tau_isTruthMatchedTau[:, 1] == 1]
    #         ar = ar[ar.tau_truthType[:, 1] == 10]

    return ar


def _apply_kinematic_cuts(ar: ak.Array, scope: str, subject: str | None) -> ak.Array:
    """Apply kinematic selection cuts on jets, MET, and angular variables."""
    ar = ar[ar.jet_n >= 2]
    ar = ar[ar.met / 1000 >= 200]
    ar = ar[ar.jet_pt[:, 0] / 1000 >= 120]
    ar = ar[ar.jet_pt[:, 1] / 1000 >= 20]

    if scope in ("CC", "ML"):
        if subject != "QCD":
            ar = ar[ar.jet_delPhiMet[:, 0] >= 0.4]
            ar = ar[ar.jet_delPhiMet[:, 1] >= 0.4]
    elif scope == "NTuples":
        cond_jet_0 = ar.jet_delPhiMet[:, 0] > 0.4
        cond_jet_1 = ar.jet_delPhiMet[:, 1] > 0.4
        cond_jets = np.logical_and(cond_jet_0, cond_jet_1)
        cond_met_meff = ar.met / ar.meff < 0.2
        ar = ar[np.logical_or(cond_jets, cond_met_meff)]

    return ar


def _apply_region_cuts(
    ar: ak.Array,
    region: str | None,
    channel: str,
    ntau: str,
    subject: str | None,
    sub_subject: str | None,
) -> ak.Array:
    """Dispatch to SR or CR cut functions based on the analysis region."""
    if region == "SR":
        ar = _apply_sr_cuts(ar, channel, ntau, subject)
    elif region == "CR":
        ar = _apply_cr_cuts(ar, channel, subject, sub_subject)
    # VR and Preselection: no additional cuts
    return ar


def _apply_sr_cuts(
    ar: ak.Array,
    channel: str,
    ntau: str,
    subject: str | None,
) -> ak.Array:
    """Apply signal region cuts based on channel and subject."""
    if ntau == "1":
        if subject == "compressed":
            ar = ar[ar.tau_pt[:, 0] / 1000 < 45]
            ar = ar[ar.met / 1000 > 400]
            ar = ar[ar.tau_mtMet[:, 0] / 1000 > 80]
        elif subject == "medium-mass":
            ar = ar[ar.tau_pt[:, 0] / 1000 > 45]
            ar = ar[ar.met / 1000 > 400]
            ar = ar[ar.tau_mtMet[:, 0] / 1000 > 250]
            ar = ar[ar.ht / 1000 > 1000]
        else:
            raise ValueError(f"Unsupported SR subject '{subject}' for 1-tau channel.")
    elif channel == "2":
        if subject == "compressed":
            ar = ar[ar.Mt2 / 1000 > 70]
            ar = ar[ar.ht / 1000 < 1100]
            ar = ar[ar.sumMT / 1000 > 1600]
        elif subject == "high-mass":
            ar = ar[ar.tau_mtMet_0_and_1 / 1000 > 350]
            ar = ar[ar.ht / 1000 > 1100]
        elif subject == "multibin":
            ar = ar[ar.tau_mtMet_0_and_1 / 1000 > 150]
            ar = ar[ar.ht / 1000 > 800]
            ar = ar[ar.jet_n >= 3]
        else:
            raise ValueError(f"Unsupported SR subject '{subject}' for 2-tau channel.")
    else:
        raise ValueError(f"Unsupported SR channel '{channel}'.")
    return ar


def _apply_cr_cuts(
    ar: ak.Array,
    channel: str,
    subject: str | None,
    sub_subject: str | None,
) -> ak.Array:
    """Apply control region cuts based on subject and sub_subject."""
    if subject in ("W(taunu)", "top"):
        if subject == "W(taunu)":
            ar = ar[ar.jet_n_btag == 0]
        else:
            ar = ar[ar.jet_n_btag >= 1]

        ar = ar[ar.ht / 1000 < 800]
        ar = ar[ar.met / 1000 < 300]

        if sub_subject == "kinematic":
            ar = ar[ar.tau_n == 0]
            ar = ar[ar.jet_n >= 3]
            ar = ar[ar.mu_n == 1]
            ar = ar[ar.mu_mtMet[:, 0] / 1000 < 100]
        elif sub_subject == "true-tau":
            ar = ar[ar.tau_n == 1]
            ar = ar[ar.jet_n >= 3]
            ar = ar[ar.mu_n == 0]
            ar = ar[ar.tau_mtMet[:, 0] / 1000 < 80]
        elif sub_subject == "fake-tau":
            ar = ar[ar.tau_n == 1]
            ar = ar[ar.mu_n == 1]
            ar = ar[ar.mu_mtMet[:, 0] / 1000 < 100]
            if subject == "W(tautau)":
                ar = ar[ar.Mtaumu / 1000 > 60]
        else:
            raise ValueError(f"Unsupported CR sub_subject '{sub_subject}'.")

    elif subject == "Z(nunu)":
        if channel == "1had1lep" or channel == "1":
            ar = ar[ar.mu_n == 0]
        # 1had0lep: LeptonVeto already applied, skip mu_n cut

        ar = ar[ar.tau_n == 1]
        ar = ar[ar.jet_n_btag == 0]
        ar = ar[ar.ht / 1000 < 800]
        ar = ar[ar.met / 1000 < 300]
        ar = ar[ar.tau_mtMet[:, 0] / 1000 >= 100]
        ar = ar[ar.tau_mtMet[:, 0] / 1000 < 200]
        ar = ar[ar.met / ar.meff > 0.3]
        ar = ar[ar.jet_delPhiMet[:, 0] >= 2.0]
        ar = ar[ar.tau_delPhiMet[:, 0] >= 1.0]

    elif subject == "Z(tautau)":
        ar = ar[ar.tau_n >= 2]
        ar = ar[ar.tau_charge[:, 0] * ar.tau_charge[:, 1] < 0]
        ar = ar[ar.jet_n_btag == 0]
        ar = ar[ar.ht / 1000 < 800]
        ar = ar[(ar.tau_mtMet[:, 0] / 1000 + ar.tau_mtMet[:, 1] / 1000) < 100]
        ar = ar[ar.Mt2 / 1000 < 70]

    elif subject == "QCD":
        ar = ar[ar.tau_n == 1]
        ar = ar[(ar.jet_delPhiMet[:, 0] < 0.3) | (ar.jet_delPhiMet[:, 1] < 0.3)]
        ar = ar[ar.tau_mtMet[:, 0] / 1000 > 100]
        ar = ar[ar.tau_mtMet[:, 0] / 1000 < 200]
        ar = ar[ar.met / ar.meff < 0.2]

    return ar


def _compute_weight(ar: ak.Array) -> ak.Array:
    """Compute and assign the combined event weight from all weight components."""
    ar["weight"] = (
        ar.lumiweight
        * ar.mcEventWeight
        * ar.pileupweight
        * ar.beamSpotWeight
        * ar.jvt_weight
        * ar.bjet_weight
        * ar.ele_weight
        * ar.mu_weight
    )
    return ar


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_samples(
    cfg: DictConfig,
    sample_type: str,
    sample_ids: list[str],
) -> dict[str, ak.Array]:
    """Read ROOT files, apply all selection cuts and weights, and return processed arrays keyed by sample id."""
    scope = cfg.analysis.scope
    channel = str(cfg.analysis.channel) if cfg.analysis.channel is not None else "1"
    region = cfg.analysis.get("region")
    subject = cfg.analysis.get("subject")
    sub_subject = cfg.analysis.get("sub_subject")
    ntau = _ntau_from_channel(channel)
    tree_name = cfg.paths.tree_name

    load_features = resolve_features(cfg)

    log.info(
        "Processing %s samples | region=%s | channel=%s | subject=%s | sub_subject=%s",
        sample_type,
        region,
        channel,
        subject,
        sub_subject,
    )

    data_out: dict[str, ak.Array] = {}

    campaigns = list(cfg.analysis.campaigns)
    total_files = len(sample_ids) * len(campaigns)

    with tqdm(total=total_files, desc=f"Processing {sample_type}", unit="file") as pbar:
        for sample_id in sample_ids:
            for j, campaign in enumerate(campaigns):
                pbar.set_postfix(sample=sample_id, campaign=campaign)
                path = _resolve_path(cfg, sample_type, campaign, sample_id)

                try:
                    with uproot.open(path) as root_file:
                        tree = root_file[tree_name]
                        ar = tree.arrays(load_features, library="ak")
                except FileNotFoundError:
                    log.warning("File not found, skipping: %s", path)
                    pbar.update(1)
                    continue

                ar = _apply_channel_cuts(ar, channel, ntau, region)
                ar = _apply_cleaning_cuts(ar)
                ar = _apply_rnn_cuts(ar, ntau)
                ar = _apply_truth_cuts(ar, ntau, sample_type, scope)
                ar = _apply_kinematic_cuts(ar, scope, subject)
                ar = _apply_region_cuts(ar, region, channel, ntau, subject, sub_subject)
                ar = _compute_weight(ar)

                if sample_id not in data_out:
                    data_out[sample_id] = ar
                else:
                    data_out[sample_id] = ak.concatenate(
                        [data_out[sample_id], ar], axis=0
                    )

                pbar.update(1)

    return data_out
