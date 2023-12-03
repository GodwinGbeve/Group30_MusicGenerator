import streamlit as st
import numpy as np
import pandas as pd
import pretty_midi
from tensorflow.keras.models import model_from_json
import collections
from music21 import stream, instrument, note


# Function to convert MIDI to notes
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)

    # Check if there are any instruments in the MIDI file
    if not pm.instruments:
        st.error(f"No instruments found in the MIDI file: {midi_file}")
        return pd.DataFrame()

    # Retrieve the first instrument
    instrument = pm.instruments[0]

    # Rest of your code to process the notes
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)  # Add velocity information
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


# # Function to merge MIDI files with a small overlap
# def merge_midi_files(midi_files, overlap_duration=1):
#     merged_score = stream.Score()

#     for file in midi_files:
#         midi_score = converter.parse(file)
#         tracks = midi_score.parts

#         for i, track in enumerate(tracks):
#             if i > 0:
#                 merged_score[i - 1].elements[-1].endTime += overlap_duration
#                 track.elements[0].offset += overlap_duration

#             merged_score.append(track)

#     return merged_score


# Function to convert notes to MIDI
def notes_to_midi_generated(notes_df, out_file, program_number, velocity=100):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program_number)

    for i, note in notes_df.iterrows():
        start = float(note['start'])
        end = float(note['end'])
        pitch = int(note['pitch'])
        velocity = int(note.get('velocity', 0))
        note_obj = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        instrument.notes.append(note_obj)

    pm.instruments.append(instrument)
    pm.write(out_file)


# Load the trained model architecture
with open('best_model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('best_model_weights.h5')

# Function to generate music
def generate_music(seed_input):
    # Normalize seed notes
    seed_input = seed_input / [128, 1.0, 1.0]

    # Generate music using the loaded model
    prediction = loaded_model.predict(np.expand_dims(seed_input, axis=0))
    generated_pitch = int(np.argmax(prediction['pitch']))
    generated_step = float(prediction['step'][0, 0])
    generated_duration = float(prediction['duration'][0, 0])


    return generated_pitch, generated_step, generated_duration

# Streamlit app
st.title('Music Generation App')

# Upload multiple MIDI files
uploaded_files = st.file_uploader('Upload MIDI files', type=['mid', 'midi'], accept_multiple_files=True)

if uploaded_files:
    st.write('Uploaded MIDI files details:')
    generated_notes_list = []  # List to store generated notes for each file
    
    for uploaded_file in uploaded_files:
        st.write(f'File Name: {uploaded_file.name}')
        st.write(f'File Size: {uploaded_file.size} bytes')

    # Button to generate music from the uploaded files
    if st.button('Generate Music'):
        # Process the uploaded MIDI files
        uploaded_notes_list = [midi_to_notes(uploaded_file) for uploaded_file in uploaded_files]

        # Concatenate notes from all files
        all_notes_df = pd.concat(uploaded_notes_list, ignore_index=True)

        random_input = np.random.rand(25, 3)

        # Generate music based on the concatenated notes
        generated_pitch, generated_step, generated_duration = generate_music(random_input)

        st.write(f'Generated Pitch: {generated_pitch}')
        st.write(f'Generated Step: {generated_step}')
        st.write(f'Generated Duration: {generated_duration}')

        # Example: Create a MIDI file with the generated music
        generated_notes = pd.DataFrame([[generated_pitch, generated_step, generated_duration, 0, generated_duration]],
                                       columns=['pitch', 'step', 'duration', 'start', 'end'])
        out_file = 'generated_music.mid'
        # Use a predefined mapping for common instrument names
        instrument_name_mapping = {
            'Acoustic Grand Piano': 0,
            'Bright Acoustic Piano': 1,
            'Electric Grand Piano': 2,
        # Add more instrument names and corresponding program numbers as needed
        }

        # Get the program number based on the instrument name
        program_number = instrument_name_mapping.get('Acoustic Grand Piano', 0)
        notes_to_midi_generated(generated_notes, out_file=out_file, program_number=program_number)

        # Provide a download link for the generated MIDI file
        download_button_str = f"Download the generated music: {out_file}"
        st.download_button(label=download_button_str, key="download_button", data=open(out_file, "rb").read(), file_name=out_file)

        # Display the generated music on the Streamlit app
        st.audio(open(out_file, 'rb').read(), format='audio/midi', start_time=0)