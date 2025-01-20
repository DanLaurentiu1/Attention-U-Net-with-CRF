from training.training_binary import training_binary
from training.training_multi_class import training_multi_class

if __name__ == "__main__":
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 2
    BINARY_MODEL_SAVE_PATH = "model/model_final_parameters/model_final_binary.pth"
    MULTI_CLASS_MODEL_SAVE_PATH = "model/model_final_parameters/model_final_multi_class.pth"

    # training_binary(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, model_save_path=BINARY_MODEL_SAVE_PATH)
    training_multi_class(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, model_save_path=MULTI_CLASS_MODEL_SAVE_PATH)
