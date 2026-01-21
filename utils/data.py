import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import logging

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    def __init__(self, datatype, imb_factor):
        super().__init__()
        self.datatype = datatype
        self.imb_factor = imb_factor
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        labels = self.train_targets
        if self.datatype == "conv":
            pass
        elif self.datatype == "lt":
            self.num_per_cls = {}
            for label in labels:
                if label in self.num_per_cls:
                    self.num_per_cls[label] += 1
                else:
                    self.num_per_cls[label] = 1

            img_num_per_cls = self.get_img_num_per_cls(self.num_per_cls)
            logging.info("img_num_per_cls {}".format(img_num_per_cls))

            # 1. 根据self.train_targets分类数据集中的每个样本
            classwise_data = {i: [] for i in range(10)}
            classwise_targets = {i: [] for i in range(10)}

            for idx, label in enumerate(self.train_targets):
                classwise_data[label].append(self.train_data[idx])
                classwise_targets[label].append(label)

            # 2. 根据img_num_per_cls随机选择每个类别的样本
            new_train_data = []
            new_train_targets = []

            for label, count in enumerate(img_num_per_cls):
                if count > len(classwise_data[label]):
                    raise ValueError(f"Required count {count} is more than available samples {len(classwise_data[label])} for class {label}.")
                
                chosen_indices = np.random.choice(len(classwise_data[label]), count, replace=False)
                new_train_data.extend(np.array(classwise_data[label])[chosen_indices])
                new_train_targets.extend(np.array(classwise_targets[label])[chosen_indices])

            # 3. 更新self.train_data 和 self.train_targets
            self.train_data = np.array(new_train_data)
            self.train_targets = np.array(new_train_targets)


    def get_img_num_per_cls(self, num_per_cls):
        img_max = 5000
        imb_factor = self.imb_factor
        img_num_per_cls = []
        for cls_idx in range(len(num_per_cls)):
            num = img_max * (imb_factor**(cls_idx / (len(num_per_cls)- 1.0)))
            img_num_per_cls.append(int(num))
        if self.datatype == "lt":
            np.random.seed(1993)
            np.random.shuffle(img_num_per_cls) # shuffle的长尾分布
        return img_num_per_cls


class iCIFAR100(iData):
    def __init__(self, datatype, imb_factor):
        super().__init__()
        self.datatype = datatype
        self.imb_factor = imb_factor
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/datasets", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/datasets", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        if self.datatype == "conv":
            pass
        elif self.datatype == "lt":
            labels = self.train_targets

            self.num_per_cls = {}
            for label in labels:
                if label in self.num_per_cls:
                    self.num_per_cls[label] += 1
                else:
                    self.num_per_cls[label] = 1

            img_num_per_cls = self.get_img_num_per_cls(self.num_per_cls)
            logging.info("img_num_per_cls {}".format(img_num_per_cls))

            # 1. 根据self.train_targets分类数据集中的每个样本
            classwise_data = {i: [] for i in range(10)}
            classwise_targets = {i: [] for i in range(10)}

            for idx, label in enumerate(self.train_targets):
                classwise_data[label].append(self.train_data[idx])
                classwise_targets[label].append(label)

            # 2. 根据img_num_per_cls随机选择每个类别的样本
            new_train_data = []
            new_train_targets = []

            for label, count in enumerate(img_num_per_cls):
                if count > len(classwise_data[label]):
                    raise ValueError(f"Required count {count} is more than available samples {len(classwise_data[label])} for class {label}.")
                
                chosen_indices = np.random.choice(len(classwise_data[label]), count, replace=False)
                new_train_data.extend(np.array(classwise_data[label])[chosen_indices])
                new_train_targets.extend(np.array(classwise_targets[label])[chosen_indices])

            # 3. 更新self.train_data 和 self.train_targets
            self.train_data = np.array(new_train_data)
            self.train_targets = np.array(new_train_targets)


    def get_img_num_per_cls(self, num_per_cls):
        img_max = 500
        imb_factor = self.imb_factor
        img_num_per_cls = []
        for cls_idx in range(len(num_per_cls)):
            num = img_max * (imb_factor**(cls_idx / (len(num_per_cls)- 1.0)))
            img_num_per_cls.append(int(num))
        if self.datatype == "lt":
            np.random.seed(1993)
            np.random.shuffle(img_num_per_cls) # shuffle的长尾分布
        return img_num_per_cls

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.use_path = False


        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/datasets", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/datasets", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
    
class iImageNet1000(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            ]
        self.test_trsf = [
            transforms.Resize(256),
        transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/imagenet/train/"
        test_dir = "/datasets/imagenet/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/imagenet/train/"
        test_dir = "/datasets/imagenet/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.use_path = True


        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/imagenet-r/train/"
        test_dir = "/datasets/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/imagenet-a/train/"
        test_dir = "/datasets/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/cub/train/"
        test_dir = "/datasets/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/objectnet/train/"
        test_dir = "/datasets/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/omnibenchmark/train/"
        test_dir = "/datasets/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/datasets/vtab-cil/vtab/train/"
        test_dir = "/datasets/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)