from .data_info import DataInfo

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import logging
import pandas as pd
import jsonlines

from .rw_jsonlines import read_jsonl, write_jsonl


class JsonlDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading jsonl data from {input_filename}.')

        # df = pd.read_csv(input_filename, sep=sep)
        # self.images = df[img_key].tolist()
        # self.captions = df[caption_key].tolist()
        data_list = read_jsonl(input_filename)
        self.images, self.captions = [], []
        for item in data_list:
            self.images.append(item[img_key])
            self.captions.append(item[caption_key])
        # end for

        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        # [3, 224, 224], [77]
        # raise RuntimeError(images.shape, texts.shape)
        # return images, texts
        return {'image': images, 'text': texts}



def get_jsonl_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonlDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
