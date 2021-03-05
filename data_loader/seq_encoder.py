from pathlib import Path
import gzip
from mimetypes import guess_type
from functools import partial
from torch.utils.data import Dataset

BASE_DICT = {"A": (1, 0, 0, 0),
             "C": (0, 1, 0, 0),
             "G": (0, 0, 1, 0),
             "T": (0, 0, 0, 1),
             "U": (0, 0, 0, 1)
             }

ZERO_LIST = (0, 0, 0, 0)

def seq_parser(seq_fh, seq_type):
    if seq_type == 'fastq':
        record_line = 0  # before parse seq record
        for line in seq_fh:
            line = line.rstrip()
            # if the current line is header line
            if line[0] == '@' and record_line == 0:
                record_line = 1
                header = line
            # if it is the next line of the header which is the seq reads
            elif record_line == 1:
                seq = line
                record_line = 2
            # if it is the next line of seq, which is + line
            elif record_line == 2:
                endseq = line
                record_line = 3
            # quality line
            else:
                qual = line
                yield header, seq, endseq, qual
                header = seq = endseq = qual = ''
                record_line = 0

    else:
        seq = header = ''
        for line in seq_fh:
            line = line.rstrip()
            if line[0] == '>' and seq != '':
                yield header, seq
                header = line
                seq = ''
            else:
                seq += line.upper()
        if seq != '':
            yield header, seq


def get_seq_format(seq_file):
    fa_exts = [".fasta", ".fa", ".fna", ".fas"]
    fq_exts = [".fq", ".fastq"]
    encoding = guess_type(seq_file)[1]  # uses file extension
    if encoding is None:
        encoding = ""
    elif encoding == "gzip":
        encoding = "gz"
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))
    seq_filename = Path(seq_file).stem if encoding == 'gz' else Path(seq_file).name
    seq_file_ext = Path(seq_filename).suffix
    if seq_file_ext not in (fa_exts + fq_exts):
        raise ValueError("""Unknown extension {}. Only fastq and fasta sequence formats are supported. 
And the file must end with one of ".fasta", ".fa", ".fna", ".fas", ".fq", ".fastq" 
and followed by ".gz" or ".gzip" if they are gzipped.""".format(seq_file_ext))
    seq_format = "fa" + encoding if seq_file_ext in fa_exts else "fq" + encoding
    return seq_format


def all_seqs_x(seq_file, min_seq_length):
    dataset = []
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith("gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    with _open(seq_file) as fh:
        # for record in SeqIO.parse(fh, seq_type):  # parse_seq_file(seq_file):
        #     seq = str(record.seq).upper()
        for record in seq_parser(fh, seq_type):
            features = seq_to_feature(record[1], min_seq_length)
            try:
                dataset.append(features)
            except NameError as e:
                print(NameError("Can not concatenate the np array", e))

        return dataset


def seq_to_feature(seq, min_seq_length):
    read_length = len(seq)
    if read_length > min_seq_length:
        start = (read_length - min_seq_length) // 2
        end = min_seq_length + start
        seq = seq[start:end]
    seq_feature = [BASE_DICT.get(base, ZERO_LIST) for base in seq]
    if read_length < min_seq_length:
        seq_feature.extend([ZERO_LIST] * (min_seq_length - read_length))

    return seq_feature
  
  
class SeqFeature(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)  
  
