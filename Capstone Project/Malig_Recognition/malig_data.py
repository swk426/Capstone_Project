import numpy as np

train_img_file = '../data/processed/train_img.csv'
train_labels_file = '../data/processed/train_labels.csv'

test_img_file = '../data/processed/test_img.csv'
test_labels_file = '../data/processed/test_labels.csv'

val_img_file = '../data/processed/val_img.csv'
val_labels_file = '../data/processed/val_labels.csv'

smote_img_file = '../data/processed/smote_train_img.csv'
smote_labels_file = '../data/processed/smote_train_labels.csv'

adasyn_img_file = '../data/processed/adasyn_train_img.csv'
adasyn_labels_file = '../data/processed/adasyn_train_labels.csv'

train_img_data = open(train_img_file, 'rt')
train_img = np.loadtxt(train_img_data, delimiter=',')
print("train_img shape",train_img.shape)
train_images = train_img.reshape(train_img.shape[0], 64,64,3)
print("train_images shape",train_images.shape)

train_labels_data = open(train_labels_file, 'rt')
train_labels = np.loadtxt(train_labels_data, delimiter = ',')
print("train_labels shape",train_labels.shape)

test_img_data = open(test_img_file, 'rt')
test_img = np.loadtxt(test_img_data, delimiter=',')
print('test_img shape', test_img.shape)
test_images = test_img.reshape(test_img.shape[0], 64,64,3)
print('test_images shape', test_images.shape)

test_labels_data = open(test_labels_file, 'rt')
test_labels = np.loadtxt(test_labels_data, delimiter=',')
print('test_labels shape', test_labels.shape)

val_img_data = open(val_img_file, 'rt')
val_img = np.loadtxt(val_img_data, delimiter=',')
print('val_img shape',val_img.shape)
val_images = val_img.reshape(val_img.shape[0],64,64,3)
print('val_images shape',val_images.shape)

val_labels_data = open(val_labels_file, 'rt')
val_labels = np.loadtxt(val_labels_data, delimiter=',')
print('val_labels shape',val_labels.shape)

smote_img_data = open(smote_img_file, 'rt')
smote_img = np.loadtxt(smote_img_data, delimiter=',')
print('smote_img shape',smote_img.shape)
smote_images = smote_img.reshape(smote_img.shape[0],64,64,3)
print('smote_images shape',smote_images.shape)

smote_labels_data = open(smote_labels_file, 'rt')
smote_labels = np.loadtxt(smote_labels_data, delimiter = ',')
print('smote_labels shape',smote_labels.shape)

adasyn_img_data = open(adasyn_img_file, 'rt')
adasyn_img = np.loadtxt(adasyn_img_data, delimiter=',')
print('adasyn_img shape',adasyn_img.shape)
adasyn_images = adasyn_img.reshape(adasyn_img.shape[0],64,64,3)
print('adasyn_images shape',adasyn_images.shape)

adasyn_labels_data = open(adasyn_labels_file, 'rt')
adasyn_labels = np.loadtxt(adasyn_labels_data, delimiter = ',')
print('adasyn_labels shape',adasyn_labels.shape)

train_y = np.reshape(train_labels[:,0], (train_img.shape[0],1))
print('train_y shape', train_y.shape)
test_y = np.reshape(test_labels[:,0], (test_img.shape[0],1))
print('test_y shape', test_y.shape)
val_y = np.reshape(val_labels[:,0], (val_img.shape[0],1))
print('val_y shape', val_y.shape)