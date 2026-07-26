these are notes from my experimentsn, learning etc..

in the reasoning from scratch book , we use max_new_tokens=2048 (chapter 3 MATH500). the overall accuracy there was around 25% but it seems that it goes in circles less than this current baseline.ipynb.
I feel that the current baseline is struggling because :
- post-trainign distribution is not good enough:  qwen3 is most likely trained on math and code so it knwos hen to stop generatino 
- harder input ? FEN is hard to decode (even when including the ascii board)