#%%
from keras.preprocessing.image import ImageDataGenerator
#%%
train_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#%%
test_gen = ImageDataGenerator(rescale=1./255)
#%%
train_set = train_gen.flow_from_directory(
        'data/train', 
        target_size=(100, 100),
        batch_size=25,
        class_mode='binary')
#%%
test_set = test_gen.flow_from_directory(
        'data/test',
        target_size=(100, 100),
        batch_size=25,
        class_mode='binary')
#%%

#%%

#%%

#%%

#%%

#%%

#%%