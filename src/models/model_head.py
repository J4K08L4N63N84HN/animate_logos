import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes, out_size):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

        # Output Layers
        self.out = nn.Linear(hidden_sizes[-1], out_size)

    # Forward Pass
    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.out(x))
        return output


def transform_binary_model_output(output):
    """ Function to translate the binary model output to animation commands

    Example: transform_binary_model_output(np.array([1,0,0,1,0,0,0,1,1,0,1,0,1,1,1,0,1,0,1,0,0]))

    Args:
        output (np.array): 21-dimensional list of binary values

    """
    output = (output > 0.5).int()

    if output[0] == 0 and output[1] == 0:
        type = 'translate'
    elif output[0] == 1 and output[1] == 1:
        type = 'scale'
    elif output[0] == 1 and output[1] == 0:
        type = 'rotate'
    elif output[0] == 0 and output[1] == 1:
        type = 'skewX'

    if output[2] == 0 and output[3] == 0:
        begin = 0
    elif output[2] == 1 and output[3] == 1:
        begin = 1
    elif output[2] == 1 and output[3] == 0:
        begin = 3
    elif output[2] == 0 and output[3] == 1:
        begin = 5

    if output[4] == 0 and output[5] == 0:
        dur = 1
    elif output[4] == 1 and output[5] == 1:
        dur = 3
    elif output[4] == 1 and output[5] == 0:
        dur = 5
    elif output[4] == 0 and output[5] == 1:
        dur = 10

    if output[6] == 0 and output[7] == 0:
        repeatCount = 1
    elif output[6] == 1 and output[7] == 1:
        repeatCount = 3
    elif output[6] == 1 and output[7] == 0:
        repeatCount = 5
    elif output[6] == 0 and output[7] == 1:
        repeatCount = 'indefinite'

    if output[8] == 1:
        fill = 'freeze'
    elif output[8] == 0:
        fill = 'remove'

    if output[9] == 0 and output[10] == 0 and output[11] == 0:
        if type == 'translate':
            from_ = -80
        elif type == 'scale':
            from_ = -2
        elif type == 'rotate':
            from_ = -180
        elif type == 'skewX':
            from_ = -180
    elif output[9] == 1 and output[10] == 1 and output[11] == 1:
        if type == 'translate':
            from_ = -60
        elif type == 'scale':
            from_ = -1
        elif type == 'rotate':
            from_ = -90
        elif type == 'skewX':
            from_ = -90
    elif output[9] == 1 and output[10] == 0 and output[11] == 0:
        if type == 'translate':
            from_ = -40
        elif type == 'scale':
            from_ = -0.5
        elif type == 'rotate':
            from_ = -45
        elif type == 'skewX':
            from_ = -45
    elif output[9] == 1 and output[10] == 1 and output[11] == 0:
        if type == 'translate':
            from_ = -20
        elif type == 'scale':
            from_ = 0
        elif type == 'rotate':
            from_ = 0
        elif type == 'skewX':
            from_ = 0
    elif output[9] == 0 and output[10] == 0 and output[11] == 1:
        if type == 'translate':
            from_ = 0
        elif type == 'scale':
            from_ = 0.5
        elif type == 'rotate':
            from_ = 45
        elif type == 'skewX':
            from_ = 45
    elif output[9] == 0 and output[10] == 1 and output[11] == 0:
        if type == 'translate':
            from_ = 20
        elif type == 'scale':
            from_ = 1
        elif type == 'rotate':
            from_ = 90
        elif type == 'skewX':
            from_ = 90
    elif output[9] == 0 and output[10] == 1 and output[11] == 1:
        if type == 'translate':
            from_ = 40
        elif type == 'scale':
            from_ = 1.5
        elif type == 'rotate':
            from_ = 180
        elif type == 'skewX':
            from_ = 180
    elif output[9] == 1 and output[10] == 0 and output[11] == 1:
        if type == 'translate':
            from_ = 60
        elif type == 'scale':
            from_ = 2
        elif type == 'rotate':
            from_ = 360
        elif type == 'skewX':
            from_ = 360

    if output[12] == 0 and output[13] == 0 and output[14] == 0:
        if type == 'translate':
            to = -60
        elif type == 'scale':
            to = -2
        elif type == 'rotate':
            to = -180
        elif type == 'skewX':
            to = -180
    elif output[12] == 1 and output[13] == 1 and output[14] == 1:
        if type == 'translate':
            to = -40
        elif type == 'scale':
            to = -1
        elif type == 'rotate':
            to = -90
        elif type == 'skewX':
            to = -90
    elif output[12] == 1 and output[13] == 0 and output[14] == 0:
        if type == 'translate':
            to = -20
        elif type == 'scale':
            to = -0.5
        elif type == 'rotate':
            to = -45
        elif type == 'skewX':
            to = -45
    elif output[12] == 1 and output[13] == 1 and output[14] == 0:
        if type == 'translate':
            to = 0
        elif type == 'scale':
            to = 0
        elif type == 'rotate':
            to = 0
        elif type == 'skewX':
            to = 0
    elif output[12] == 0 and output[13] == 0 and output[14] == 1:
        if type == 'translate':
            to = 20
        elif type == 'scale':
            to = 0.5
        elif type == 'rotate':
            to = 45
        elif type == 'skewX':
            to = 45
    elif output[12] == 0 and output[13] == 1 and output[14] == 0:
        if type == 'translate':
            to = 40
        elif type == 'scale':
            to = 1
        elif type == 'rotate':
            to = 90
        elif type == 'skewX':
            to = 90
    elif output[12] == 0 and output[13] == 1 and output[14] == 1:
        if type == 'translate':
            to = 60
        elif type == 'scale':
            to = 1.5
        elif type == 'rotate':
            to = 180
        elif type == 'skewX':
            to = 180
    elif output[12] == 1 and output[13] == 0 and output[14] == 1:
        if type == 'translate':
            to = 80
        elif type == 'scale':
            to = 2
        elif type == 'rotate':
            to = 360
        elif type == 'skewX':
            to = 360

    if output[15] == 0 and output[16] == 0 and output[17] == 0:
        if type == 'translate':
            fromY = -80
        elif type == 'scale':
            fromY = -2
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 1 and output[16] == 1 and output[17] == 1:
        if type == 'translate':
            fromY = -60
        elif type == 'scale':
            fromY = -1
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 1 and output[16] == 0 and output[17] == 0:
        if type == 'translate':
            fromY = -40
        elif type == 'scale':
            fromY = -0.5
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 1 and output[16] == 1 and output[17] == 0:
        if type == 'translate':
            fromY = -20
        elif type == 'scale':
            fromY = 0
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 0 and output[16] == 0 and output[17] == 1:
        if type == 'translate':
            fromY = 0
        elif type == 'scale':
            fromY = 0.5
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 0 and output[16] == 1 and output[17] == 0:
        if type == 'translate':
            fromY = 20
        elif type == 'scale':
            fromY = 1
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 0 and output[16] == 1 and output[17] == 1:
        if type == 'translate':
            fromY = 40
        elif type == 'scale':
            fromY = 1.5
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None
    elif output[15] == 1 and output[16] == 0 and output[17] == 1:
        if type == 'translate':
            fromY = 60
        elif type == 'scale':
            fromY = 2
        elif type == 'rotate':
            fromY = None
        elif type == 'skewX':
            fromY = None

    if output[18] == 0 and output[19] == 0 and output[20] == 0:
        if type == 'translate':
            toY = -60
        elif type == 'scale':
            toY = -2
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 1 and output[19] == 1 and output[20] == 1:
        if type == 'translate':
            toY = -40
        elif type == 'scale':
            toY = -1
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 1 and output[19] == 0 and output[20] == 0:
        if type == 'translate':
            toY = -20
        elif type == 'scale':
            toY = -0.5
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 1 and output[19] == 1 and output[20] == 0:
        if type == 'translate':
            toY = 0
        elif type == 'scale':
            toY = 0
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 0 and output[19] == 0 and output[20] == 1:
        if type == 'translate':
            toY = 20
        elif type == 'scale':
            toY = 0.5
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 0 and output[19] == 1 and output[20] == 0:
        if type == 'translate':
            toY = 40
        elif type == 'scale':
            toY = 1
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 0 and output[19] == 1 and output[20] == 1:
        if type == 'translate':
            toY = 60
        elif type == 'scale':
            toY = 1.5
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None
    elif output[18] == 1 and output[19] == 0 and output[20] == 1:
        if type == 'translate':
            toY = 80
        elif type == 'scale':
            toY = 2
        elif type == 'rotate':
            toY = None
        elif type == 'skewX':
            toY = None

    return type, begin, dur, repeatCount, fill, from_ , to, fromY, toY
