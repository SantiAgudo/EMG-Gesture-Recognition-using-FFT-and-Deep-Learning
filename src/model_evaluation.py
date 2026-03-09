# Model Evaluation and Saving

loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {accuracy:.2f}')

# Save the trained model
model.save('/content/drive/My Drive/PAPER/movement_model.h5')
