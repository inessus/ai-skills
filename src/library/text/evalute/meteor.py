import re
import os
from os.path import join
import tempfile
import subprocess as sp

from cytoolz import curry


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None


def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir):
    """
         METEOR evaluation
    :param dec_pattern:
    :param dec_dir:
    :param ref_pattern:
    :param ref_dir:
    :return:
    """
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))
    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(join(tmp_dir, 'ref.txt'), 'w') as ref_f,\
             open(join(tmp_dir, 'dec.txt'), 'w') as dec_f:
            ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')
            dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

        cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
            _METEOR_PATH, join(tmp_dir, 'dec.txt'), join(tmp_dir, 'ref.txt'))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
