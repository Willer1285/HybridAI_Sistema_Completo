import onnxruntime as rt
sess = rt.InferenceSession("modelo/hybrid_ai_model.onnx")
out = sess.get_outputs()[0]
print("Shape is:", out.shape)
