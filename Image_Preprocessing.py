import pandas as pd, numpy as np, matplotlib.pyplot as plt
import shutil, time, os
import keras, sklearn

#######################################################

print("\nCreating Processed Data Directories")

base = 'Processed Data'; os.mkdir(base)

os.mkdir(os.path.join(base, "Training"))

os.mkdir(os.path.join(base, "Training", 'nv'))
os.mkdir(os.path.join(base, "Training", 'mel'))
os.mkdir(os.path.join(base, "Training", 'bkl'))
os.mkdir(os.path.join(base, "Training", 'bcc'))
os.mkdir(os.path.join(base, "Training", 'akiec'))
os.mkdir(os.path.join(base, "Training", 'vasc'))
os.mkdir(os.path.join(base, "Training", 'df'))

os.mkdir(os.path.join(base, "Validation"))

os.mkdir(os.path.join(base, "Validation", 'nv'))
os.mkdir(os.path.join(base, "Validation", 'mel'))
os.mkdir(os.path.join(base, "Validation", 'bkl'))
os.mkdir(os.path.join(base, "Validation", 'bcc'))
os.mkdir(os.path.join(base, "Validation", 'akiec'))
os.mkdir(os.path.join(base, "Validation", 'vasc'))
os.mkdir(os.path.join(base, "Validation", 'df'))

#######################################################

print("\nReading Original HAM10000 Metadata")

df_data = pd.read_csv('Original Data/HAM10000_metadata.csv')
print(df_data.head())

#######################################################

print("\nSorting Data Labels Based on Pre-Augmented Duplicates")

df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)

#######################################################

print("\nTemporarily Removing Duplicates to Sift Validation Data\n")

def identify_duplicates(x):  return 'no_duplicates' if x in list(df['lesion_id']) else 'has_duplicates'
    
df_data['duplicates'] = df_data['lesion_id']
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

print(df_data.head())

print("\n" + str(df_data['duplicates'].value_counts()))
df = df_data[df_data['duplicates'] == 'no_duplicates']

#######################################################

print("\nSplitting Dataset into Training (83%)/Testing (17%) Partitions")

_, df_val = sklearn.model_selection.train_test_split(df, test_size=0.17, random_state=101, stratify=df['dx'])
df_val = df_val.drop('duplicates', axis=1)

#######################################################

print("\nIdentifying Data Records in Overall Set Not in Validation to be in Training")

def identify_val_rows(x): return 'val' if str(x) in list(df_val['image_id']) else 'train'

df_data['train_or_val'] = df_data['image_id']
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)

df_train = df_data[df_data['train_or_val'] == 'train']

df_train = df_train.drop('train_or_val', axis=1)
df_train = df_train.drop('duplicates', axis=1)

#######################################################

print("\n2 Distinct Training & Testing Datasets Created!")

print("\nTraining Classes Count:\n"); print(df_train['dx'].value_counts())
print("\nValidation Classes Count:\n"); print(df_val['dx'].value_counts())

print("\nShapes of Training & Validation Dataframes\n")
print(df_train.shape); print(df_val.shape)

print("\nSaving Partitions to CSVs\n")
df_train.to_csv(path_or_buf='Processed Data/Training/df_train.csv', index=False); print('df_train.csv')
df_val.to_csv(path_or_buf='Processed Data/Validation/df_val.csv', index=False); print('df_val.csv')

#######################################################

print("\nTransferring Images Based on Seperate Classes...")

df_data.set_index('image_id', inplace=True)

folder_1 = os.listdir('Original Data/ham10000_images_part_1'); train_list = list(df_train['image_id'])
folder_2 = os.listdir('Original Data/ham10000_images_part_2'); val_list = list(df_val['image_id'])

for image in train_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        src = os.path.join('Original Data/ham10000_images_part_1', fname)
        dst = os.path.join(base, "Training", label, fname)
        shutil.copyfile(src, dst)
    if fname in folder_2:
        src = os.path.join('Original Data/ham10000_images_part_2', fname)
        dst = os.path.join(base, "Training", label, fname)
        shutil.copyfile(src, dst)

print("\nTraining Images per Class Folder\n")
print("nv: ", end=''); print(len(os.listdir(base + '/Training/nv')))
print("mel: ", end=''); print(len(os.listdir(base + '/Training/mel')))
print("bkl: ", end=''); print(len(os.listdir(base + '/Training/bkl')))
print("bcc: ", end=''); print(len(os.listdir(base + '/Training/bcc')))
print("akiec: ", end=''); print(len(os.listdir(base + '/Training/akiec')))
print("vasc: ", end=''); print(len(os.listdir(base + '/Training/vasc')))
print("df: ", end=''); print(len(os.listdir(base + '/Training/df')))

for image in val_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        src = os.path.join('Original Data/ham10000_images_part_1', fname)
        dst = os.path.join(base, "Validation", label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('Original Data/ham10000_images_part_2', fname)
        dst = os.path.join(base, "Validation", label, fname)
        shutil.copyfile(src, dst)
        
print("\nValidation Images per Class Folder\n")
print("nv: ", end=''); print(len(os.listdir(base + '/Validation/nv')))
print("mel: ", end=''); print(len(os.listdir(base + '/Validation/mel')))
print("bkl: ", end=''); print(len(os.listdir(base + '/Validation/bkl')))
print("bcc: ", end=''); print(len(os.listdir(base + '/Validation/bcc')))
print("akiec: ", end=''); print(len(os.listdir(base + '/Validation/akiec')))
print("vasc: ", end=''); print(len(os.listdir(base + '/Validation/vasc')))
print("df: ", end=''); print(len(os.listdir(base + '/Validation/df')))

#######################################################

print("\nAugmenting More of All Training Classes Except nV (Already too many!)...\n")

class_list = ['mel','bkl','bcc','akiec','vasc','df']

for img_class in class_list:
    
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    img_list = os.listdir(os.path.join(base, "Training", img_class))

    for fname in img_list: shutil.copyfile(os.path.join(base, "Training", img_class, fname), os.path.join(img_dir, fname))

    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=180, width_shift_range=0.1, 
                                                           height_shift_range=0.1, zoom_range=0.1, 
                                                           horizontal_flip=True, vertical_flip=True, 
                                                           brightness_range=(0.9,1.1), fill_mode='nearest')
    batch_size = 50
    aug_datagen = datagen.flow_from_directory(aug_dir, save_to_dir = os.path.join(base, "Training", img_class), 
                                              save_format='jpg', target_size=(224,224), batch_size=batch_size)
    
    num_aug_images_wanted = 6000
    num_batches = int(np.ceil((num_aug_images_wanted - len(os.listdir(img_dir)))/batch_size))
    for i in range(0, num_batches): imgs, labels = next(aug_datagen)
    
    time.sleep(2)    
    shutil.rmtree('aug_dir')
    
print("\nTotal Training Images per Class Folder After Augmentations!\n")
print("nv: ", end=''); print(len(os.listdir(base + '/Training/nv')))
print("mel: ", end=''); print(len(os.listdir(base + '/Training/mel')))
print("bkl: ", end=''); print(len(os.listdir(base + '/Training/bkl')))
print("bcc: ", end=''); print(len(os.listdir(base + '/Training/bcc')))
print("akiec: ", end=''); print(len(os.listdir(base + '/Training/akiec')))
print("vasc: ", end=''); print(len(os.listdir(base + '/Training/vasc')))
print("df: ", end=''); print(len(os.listdir(base + '/Training/df')))
print("\nClass Distribution Balanced!\n")

#######################################################

print("\nSome Sample Random Images")
    
def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None): # 12,6
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
plots(imgs)

#######################################################