# -*- coding: UTF-8 -*-
from agents.base import SequentialFineTune
from agents.er import ExperienceReplay
from agents.ewc import EWC
from agents.lwf import LwF
from agents.mas import MAS
from agents.si import SI
from agents.dt2w import DT2W
from agents.aser import ASER
from agents.herding import Herding
from agents.inversion import Inversion
from agents.clops import CLOPS
from agents.der import DarkExperienceReplay
from agents.gr import GenerativeReplay
from agents.er_sub import ER_on_Subject
from agents.fast_icarl import FastICARL
from agents.sleep_agent import SleepAgent
from agents.sleep_mixin_v2 import make_sleep_cl_agent
from agents.SRC import SleepReplayConsolidation
from agents.wscl import WSCL
from agents.siesta import SIESTA


# ---------------------------------------------------------------------------
# Base agent classes available as sleep-CL foundations
# (sleep-CL = SleepCLMixin + one of these, selected via --sleep_base_agent)
# ---------------------------------------------------------------------------
_BASE_AGENTS_FOR_SLEEP = {
    'ER':       ExperienceReplay,
    'EWC':      EWC,
    'LwF':      LwF,
    'SI':       SI,
    'MAS':      MAS,
    'ASER':     ASER,
    'DER':      DarkExperienceReplay,
    'CLOPS':    CLOPS,
    'GR':       GenerativeReplay,
}


# ---------------------------------------------------------------------------
# Standard agent registry
# ---------------------------------------------------------------------------
agents = {
    'ER':       ExperienceReplay,
    'EWC':      EWC,
    'LwF':      LwF,
    'SI':       SI,
    'MAS':      MAS,
    'ASER':     ASER,
    'CLOPS':    CLOPS,
    'DER':      DarkExperienceReplay,
    'GR':       GenerativeReplay,
    'sleep-CL': make_sleep_cl_agent,
    # SRC: Sleep Replay Consolidation (SNN-STDP, from sleepnn_old.m)
    'SRC':      SleepReplayConsolidation,
    # WSCL: Wake-Sleep Consolidated Learning (ER-ACE + NREM EWC sleep)
    'WSCL':     WSCL,
    # SIESTA: Sleep-Inspired Experience Sampling and Training Architecture
    'SIESTA':   SIESTA,
}

agents_replay = [
    'ER', 'DER', 'ASER','CLOPS', 'GR', 
]
