
# coding: utf-8

# # Dataset split

# In[1]:


from __future__ import division

import numpy as np
import ntcir
import ntcir.IO as IO
import os
import os.path as osp
import itertools
import utils
import shutil
from collections import defaultdict
from easydict import EasyDict as edict


# In[2]:


users = IO.load_annotations(ntcir.filepaths)
sorted_users = ntcir.utils.sort(users)
categories = IO.load_categories(ntcir.filepaths)
users_ids = sorted(users.keys())

days = defaultdict(lambda: defaultdict(ntcir.Day))
for user in sorted_users:
    for day in user.days:
        days[user.id_][day.date] = day

splits = edict({'train': 0, 'validation': 1, 'test': 2})


# # Classification dataset split

# In[ ]:


images = defaultdict(list)
targets = defaultdict(list)
for user in sorted_users:
    for day in user.days:
        for img in day.images:
            images[user.id_].append(img)
            targets[user.id_].append(img.label)


# In[2]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

folds = defaultdict(lambda: defaultdict(dict))

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.111111, random_state=42)
for user_id in users_ids:
    for i, (training_index, test_index) in enumerate(skf.split(images[user_id], targets[user_id])):
        training_targets = [images[user_id][j].label for j in training_index]

        # Compensating number of training examples
        orig_num_train_targets = len(training_targets)
        counts = np.bincount(training_targets)
        less_than_two_indices = np.nonzero(counts < 2)[0]
        for ind in less_than_two_indices:
            for j in range(2 - counts[ind]):
                training_targets.append(ind)

        train_index, val_index = sss.split(np.zeros(len(training_targets)), training_targets).next()
        folds[i][user_id][splits.train] = [images[user_id][j] for j in train_index if j < orig_num_train_targets]
        folds[i][user_id][splits.validation] = [images[user_id][j] for j in val_index if j < orig_num_train_targets]
        folds[i][user_id][splits.test] = [images[user_id][j] for j in test_index]


# In[ ]:


num_categories = len(categories)
num_images = sum([u.num_images for u in sorted_users])
padding_zeros = utils.num_digits(num_images)

for i, fold in enumerate(folds.itervalues()):

    split = [[], [], []]
    for user_id, split_ind in itertools.product(users_ids, splits.itervalues()):
        split[split_ind].extend(fold[user_id][split_ind])

    for split_name, split_id in splits.iteritems():

        split_dir = osp.join('data', 'static', str(i + 1).zfill(2), split_name)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)

        for j in xrange(num_categories):
            category = str(j).zfill(utils.num_digits(num_categories))
            category_dir = os.path.join(split_dir, category)
            utils.makedirs(category_dir)

        targets = list()
        img_paths = list()
        for image in split[split_id]:
            targets.append(image.label)
            img_paths.append(image.path)

        utils.link_images(num_categories, split_dir, padding_zeros, targets, img_paths)

