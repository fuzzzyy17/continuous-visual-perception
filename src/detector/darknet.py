from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):

    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # lines as list
    lines = [x for x in lines if len(x) > 0] # remove empty lines
    lines = [x for x in lines if x[0] != '#'] # remove comments
    lines = [x.rstrip().lstrip() for x in lines] # remove whitespace

    block = {}
    blocks = []

    for line in lines:
        if lines[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, val = line.split("=")
            block[key.rstrip()] = val.lstrip()
    blocks.append(block)

    return blocks