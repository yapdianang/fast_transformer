import os
import io
from torchtext import data

class CustomDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, fields, **kwargs):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = (os.path.expanduser(path + "/articles.txt"), os.path.expanduser(path+"/summaries.txt"))

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(CustomDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, path=None, root='data',
               train='train', validation='val', test='test', **kwargs):

        if path is None:
            path = root
            train = 'val' # Only use subset of data for now

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

# ============================================================================
