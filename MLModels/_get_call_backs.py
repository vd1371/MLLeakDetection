from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def _get_call_backs(*args, **kwargs):
		
		should_checkpoint = kwargs.get('should_checkpoint')
		should_early_stop = kwargs.get('should_early_stop')

		call_back_list = []
		
		if should_checkpoint:
			model_path = f"./models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
			checkpoint = ModelCheckpoint(model_path, 
										 monitor = 'val_acc',
										 verbose = 1,
										 save_best_only = True,
										 mode = 'max')
			call_back_list.append(checkpoint)

		if should_early_stop:
			early_stop = EarlyStopping(monitor = 'val_loss',
									   patience = self.patience,
									   verbose = 1)
			call_back_list.append(early_stop)

		return call_back_list