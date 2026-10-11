# Dummy CMSSW configuration.
#
# CRAB requires a parameter set even when the real work is done by a
# scriptExe, which is the case here: crab_script.sh runs the GAE pipeline
# scripts. This process reads the input files CRAB assigns to the job and
# does nothing else.

import FWCore.ParameterSet.Config as cms

process = cms.Process("GAE")

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(),
)
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(-1),
)
process.options = cms.untracked.PSet()
