import torch
import torchvision
import torch.nn as nn# `import torchvision.transforms as T` is importing the module `transforms` from
# the `torchvision` library and aliasing it as `T`. This allows you to use the
# functions and classes from the `transforms` module using the alias `T`, making
# the code more concise and readable.

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
to_tensor = T.ToTensor()
import typing as tp
import os

class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames и captions должны быть одинаковой длины.
        Если у изображения несколько подписей → дублируем image_filenames.
        :tokenizer -- токенизатор (например, HuggingFace)
        """
        self.max_tokenizer_length = 200
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer

        # заранее токенизируем
        self.encoded_captions = self.tokenizer(
            self.captions,
            padding=True,
            truncation=True,
            max_length=self.max_tokenizer_length,
            return_tensors="pt"
        )

        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:
        item = {
            key: values[idx] for key, values in self.encoded_captions.items()
        }

        img_path = os.path.join(self.image_path, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        item['image'] = image
        item['caption'] = self.captions[idx]  # строка (для отладки/оценки)

        return item

    def __len__(self):
        return len(self.captions)
