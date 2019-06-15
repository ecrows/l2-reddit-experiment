from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file("gs://redbert/validation-models/1560486820809_unmasked_seed44_fold1of10/model.ckpt-500", tensor_name='output_weights', all_tensors=False)
