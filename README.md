# Non-stationary Time-aware Kernelized Attention for Temporal Event Prediction
Code for our paper titled "Non-stationary Time-aware Kernelized Attention for Temporal Event Prediction" (KDD2022)

## How to use
```
encoder_model = EncoderWithGSMK()
enc_outputs = encoder_model(enc_event_inputs, time_inputs=(enc_time_inputs, enc_time_inputs))

decoder_model = DecoderWithGSMK()
dec_outputs = decoder_model(dec_event_inputs, dec_time_inputs=(dec_time_inputs, dec_time_inputs),
    cross_time_inputs=(dec_time_inputs, enc_time_inputs), enc_output=enc_outputs)
```

## Setup
- keras==2.2.4
- tensorflow==1.15.0
