# Live speech, captured

Backend: `speechmatics` · live: **True** · model `enhanced` · endpoint `wss://eu2.rt.speechmatics.com/v2`

Three clips streamed at wall-clock speed through the Speechmatics realtime WebSocket API. Every line below is in `results/voice_transcript.jsonl`.

- events: {'partial': 22, 'final': 11, 'barge_in': 4}
- finals with a measured latency: 11
- final latency: median **922 ms**, min 375 ms, max 1609 ms

Latency is measured from the moment the audio chunk covering a phrase was sent to the moment its final transcript arrived. It slightly overstates the true figure and cannot see network buffering; it is an approximation, not a lab measurement.

## Transcript

| clip | t | kind | text | latency |
| --- | --- | --- | --- | --- |
| set_the_table.wav | 2.08s | final | Set the  | 859 ms |
| set_the_table.wav | 3.21s | final | table  | 1609 ms |
| set_the_table.wav | 3.65s | final | for one,  | 1250 ms |
| set_the_table.wav | 4.40s | final | please.  | 500 ms |
| pour_water.wav | 2.13s | final | Power  | 1047 ms |
| pour_water.wav | 2.50s | final | me some  | 969 ms |
| pour_water.wav | 3.19s | final | water.  | 375 ms |
| stop_command.wav | 2.19s | final **(stop)** | Stop  | 922 ms |
| stop_command.wav | 2.19s | barge_in **(stop)** | Stop  |  |
| stop_command.wav | 3.59s | final | . Put the  | 812 ms |
| stop_command.wav | 3.59s | barge_in | . Put the  |  |
| stop_command.wav | 4.36s | final | plate  | 1125 ms |
| stop_command.wav | 4.36s | barge_in | plate  |  |
| stop_command.wav | 5.34s | final | down first.  | 469 ms |
| stop_command.wav | 5.34s | barge_in | down first.  |  |
