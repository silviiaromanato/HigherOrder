import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import IPython.display as ipd
import moviepy.editor as mp
from pydub import AudioSegment
from PIL import Image
import scipy.io.wavfile as wavfile
import sys

def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def FrameCapture(MOVIE_PATH): # IMAGES
    # Path to video file
    video = cv2.VideoCapture(MOVIE_PATH)

    count = 0
    success = 1

    average_brightness_left = []
    average_saturation_left = []
    average_hue_left = []
    average_brightness_right = []
    average_saturation_right = []
    average_hue_right = []

    while success:
        try:
            success, image = video.read()

            if success is False:
                print("Error: Unable to load the image.")
            else:
                                # divide the image in two parts: left and right
                image_left = image[:, :int(image.shape[1]/2)]
                image_right = image[:, int(image.shape[1]/2):]
                count += 1
                if count % 500 == 0:
                    print(f"Frame {count} processed")

                ############ left image ############
                h_channel, s_channel, v_channel = convert_to_hsv(image_left)        
                sum_v = np.sum(v_channel)
                sum_s = np.sum(s_channel)
                sum_h = np.sum(h_channel)
                width, height = v_channel.shape
                total_pixels = width * height

                average_brightness_left.append(sum_v / total_pixels)
                average_saturation_left.append(sum_s / total_pixels)
                average_hue_left.append(sum_h / total_pixels)

                ############ right image ############
                h_channel, s_channel, v_channel = convert_to_hsv(image_right)
                sum_v = np.sum(v_channel)
                sum_s = np.sum(s_channel)
                sum_h = np.sum(h_channel)
                width, height = v_channel.shape
                total_pixels = width * height

                average_brightness_right.append(sum_v / total_pixels)
                average_saturation_right.append(sum_s / total_pixels)
                average_hue_right.append(sum_h / total_pixels)

        except Exception as e:
            print(f"An error occurred: {e}")

    print('The average saturation of the movie is: ', np.mean(average_brightness_left))
    print('The average brightness of the movie is: ', np.mean(average_saturation_left))
    print('The average hue of the movie is: ', np.mean(average_hue_left))
    print('The average red of the movie is: ', np.mean(average_brightness_right))
    print('The average green of the movie is: ', np.mean(average_saturation_right))
    print('The average blue of the movie is: ', np.mean(average_hue_right))

    # convert the average brightness and saturation to a dataframe
    df_movie = pd.DataFrame({'average_brightness_left': average_brightness_left, 'average_saturation_left': average_saturation_left,
                            'average_hue_left': average_hue_left, 'average_brightness_right': average_brightness_right,
                            'average_saturation_right': average_saturation_right, 'average_hue_right': average_hue_right})

    return df_movie

if __name__ == '__main__':   

    movie_name = sys.argv[1]
    PATH_MOVIES = sys.argv[2]
    print('The movie name is: ', movie_name)
    print('The path of the movies is: ', PATH_MOVIES)

    MOVIE_PATH = PATH_MOVIES + movie_name
    print(MOVIE_PATH)

    #################### IMAGES ####################
    #df_movie = FrameCapture(MOVIE_PATH)
    #print(df_movie.head(30))
    #df_movie.to_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/movie_features_{movie_name[:-4]}.csv', index=False)
        
    ##################### AUDIO EXTRACTION #####################
    print('Extracting the audio from the movie...')
    movie_name = movie_name[:-4]
    print(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio{movie_name[:-4]}.wav')
    if not os.path.exists(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio{movie_name[:-4]}.wav'):
        video = mp.VideoFileClip(MOVIE_PATH)
        video_duration = video.duration
        start_time = 1  # Start time in seconds
        end_time = video_duration  # End time, limited to video duration
        clip = video.subclip(start_time, end_time)
        clip.audio.write_audiofile(f"/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio{movie_name}.wav")
        filename = f"/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio{movie_name}.wav"
        print('The audio was extracted!')

    else:
        filename = f"/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio{movie_name}.wav"
        print('The audio was already extracted and was reloaded!')

    ##################### READING THE AUDIO #####################
    # Load the audio file
    y, sr = librosa.load(filename, sr=22050)
    duration_seconds, duration_minutes = librosa.get_duration(y=y, sr=sr), librosa.get_duration(y=y, sr=sr) / 60
    print("Duration (seconds):", duration_seconds, " -- Duration (minutes):", duration_minutes)

    ##################### SPECTOGRAM #####################

    Fs, aud = wavfile.read(filename)
    aud = aud[:,0]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
    print('The power spectrum is: ', powerSpectrum, powerSpectrum.shape)
    print('The frequencies found are: ', frequenciesFound, frequenciesFound.shape)
    print('The time is: ', time, time.shape)
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/spectogram_{movie_name}.png')
    
    sys.exit()
    ##################### RMS ################### : total magnitude of the signal, LOUDNESS OF THE SIGNAL
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)

    fig, ax = plt.subplots(figsize=(15, 6), nrows=2, sharex=True) 
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer() 
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[1]) 
    ax[1].set(title='log Power spectrogram')
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/RMS_{movie_name}.png')
    # SAVE THE RMS
    #np.save(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/RMS_{movie_name[:-4]}.npy', np.array(rms))
    print(f"RMS Energy: {rms}", 'The length is: ', rms.shape)
    # exchange the dimension of the array in order to save it as a dataframe
    rms = rms.T
    # save as a dataframe
    df_rms = pd.DataFrame(rms, columns = ['rms'])
    df_rms.reset_index(drop=True, inplace=True)

    ##################### ZERO CROSSING RATE ################### : number of times that the signal crosses the horizontal axis
    zcrs = librosa.feature.zero_crossing_rate(y)
    print(f"Zero crossing rate: {sum(librosa.zero_crossings(y))}")
    plt.figure(figsize=(15, 3)) 
    plt.plot(zcrs[0])
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/ZCR_{movie_name}.png')
    #np.save(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/ZCR_{movie_name[:-4]}.npy', np.array(zcrs))
    print(f"Zero crossing rate: {zcrs}", 'The length is: ', zcrs.shape)
    zcrs = zcrs.T
    # save as a dataframe
    df_zcrs = pd.DataFrame(zcrs, columns = ['zcrs'])
    df_zcrs.reset_index(drop=True, inplace=True)

    ##################### Mel-Frequency Cepstral Coefficients (MFCCs) ################### : is a representation of the short- term power spectrum of a sound, 
                                                                                        # based on some transformation in a Mel- scale. 
                                                                                        # It is commonly used in speech recognition as people’s voices are usually on a 
                                                                                        # certain range of frequency and different from one to another.
    mfccs = librosa.feature.mfcc(y = y, sr=sr)
    plt.figure(figsize=(15, 3)) 
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/MFCCs_{movie_name}.png')
    print(f"MFCCs: {mfccs} and the length is {mfccs.shape}")
    mfccs = mfccs.T
    df_mfccs = pd.DataFrame(mfccs, columns = ['mfccs_0', 'mfccs_1', 'mfccs_2', 'mfccs_3', 'mfccs_4',
                                                    'mfccs_5', 'mfccs_6', 'mfccs_7', 'mfccs_8', 'mfccs_9', 'mfccs_10',
                                                    'mfccs_11', 'mfccs_12', 'mfccs_13', 'mfccs_14', 'mfccs_15', 'mfccs_16',
                                                    'mfccs_17', 'mfccs_18', 'mfccs_19'])
    df_mfccs.reset_index(drop=True, inplace=True)

    ##################### CHROMA ################### : dominant keys
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(y = y, sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=(15, 3))
    img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm') 
    fig.colorbar(img, ax=ax)
    ax.set(title='Chromagram')
    plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/Chroma_{movie_name}.png')
    #np.save(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/Chroma_{movie_name[:-4]}.npy', np.array(chromagram))
    print(f"Chromagram: {chromagram}", 'The length is: ', chromagram.shape)
    # save as a dataframe
    chromagram = chromagram.T
    df_chromagram = pd.DataFrame(chromagram, columns = ['chromagram_0', 'chromagram_1', 'chromagram_2', 
                                                    'chromagram_3', 'chromagram_4', 'chromagram_5', 'chromagram_6', 'chromagram_7',
                                                    'chromagram_8', 'chromagram_9', 'chromagram_10', 'chromagram_11'])
    df_chromagram.reset_index(drop=True, inplace=True)

    ##################### TEMPOGRAM ################### : is the speed or pace of a given piece and derives directly from the average beat duration.
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    print('The onset strength is: ', oenv, oenv.shape)
    df_tempo = pd.DataFrame(oenv, columns = ['tempo'])

    ##################### CONCATENATE ALL THE FEATURES ###################
    df_features = pd.concat([df_rms, df_zcrs, df_mfccs, df_chromagram], axis=1)
    df_features.to_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/features_sound_{movie_name}.csv', index=False)

    print('\n')
    print(f'The code was run for movie {movie_name}!')
    print('\n')