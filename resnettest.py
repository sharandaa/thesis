from tensorflow.keras.models import load_model
 
# load model
model = load_model('best_resnetmodel.h5')
# summarize model.
print(model.summary())