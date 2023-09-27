import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import IPython.display as ipd
import moviepy.editor as mp

#PATH_MOVIES = '/media/miplab-nas2/Data2/Movies_Emo/FilmFiles/'
# Local
PATH_MOVIES = '/Users/silviaromanato/Desktop/ServerMIPLAB/FilmFiles/'

Local = True

def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def FrameCapture(MOVIE_PATH):
    # Path to video file
    video = cv2.VideoCapture(MOVIE_PATH)

    count = 0
    success = 1

    average_brightness = []
    average_saturation = []
    average_hue = []
    while success:
        try:
            success, image = video.read()
            if success is False:
                print("Error: Unable to load the image.")
            else:
                count += 1
                if Local:
                    cv2.imwrite("/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/frame%d.jpg" % count, image)
                else:
                    cv2.imwrite("/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/frame%d.jpg" % count, image)
                h_channel, s_channel, v_channel = convert_to_hsv(image)        
                sum_v = np.sum(v_channel)
                sum_s = np.sum(s_channel)
                sum_h = np.sum(h_channel)
                height, width = v_channel.shape[:2]
                total_pixels = width * height
                average_brightness.append(sum_v / total_pixels)
                average_saturation.append(sum_s / total_pixels)
                average_hue.append(sum_h / total_pixels)
        except Exception as e:
            print(f"An error occurred: {e}")

    print('The average saturation of the movie is: ', np.mean(average_saturation))
    print('The average brightness of the movie is: ', np.mean(average_brightness))
    print('The average hue of the movie is: ', np.mean(average_hue))

    # convert the average brightness and saturation to a dataframe
    df_movie = pd.DataFrame({'average_brightness': average_brightness, 'average_saturation': average_saturation, 
                             'average_hue': average_hue})

    # save the dataframe to a csv file
    #df_movie.to_csv('movie_features.csv', index=False)

    return df_movie

if __name__ == '__main__':   
    for movie_name in os.listdir(PATH_MOVIES):
        MOVIE_PATH = PATH_MOVIES + movie_name
        print(MOVIE_PATH)
        
        df_movie = FrameCapture(MOVIE_PATH)

        print(df_movie.head())

        # plot the average brightness and saturation and hue
        plt.figure()
        plt.plot(df_movie['average_brightness'], label='brightness')
        plt.plot(df_movie['average_saturation'], label='saturation')
        plt.plot(df_movie['average_hue'], label='hue')
        plt.legend()
        if Local:
            plt.savefig('/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/image.png')
        else:
            plt.savefig('/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/image.png')

        clip = mp.VideoFileClip(MOVIE_PATH).subclip(1, 1380)
        if Local:
            clip.audio.write_audiofile("/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Audio/audioYouAgain.wav")
            filename = "/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Audio/audioYouAgain.wav"
        else:
            clip.audio.write_audiofile("/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audioYouAgain.wav")
            filename = "/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audioYouAgain.wav"
        x, sr = librosa.load(filename, sr=22050)
        int(librosa.get_duration(x, sr) / 60)
        max_slice = 10
        window_length = max_slice * sr
        a = x[21 * window_length:22 * window_length]
        ipd.Audio(a, rate=sr)

        s_energy = np.array([sum(abs(x[i:i + window_length] ** 2)) for i in range(0, len(x), window_length)])
        print(s_energy)

        plt.hist(s_energy)
        if Local:
            plt.savefig('/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/energy.png')
        else:
            plt.savefig('/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/energy.png')

        # add the energy to the dataframe
        df_movie['energy'] = s_energy

        # save the dataframe to a csv file
        if Local:
            df_movie.to_csv('/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/movie_features.csv', index=False)
        else:
            df_movie.to_csv('/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/movie_features.csv', index=False)

        print(df_movie.head())
        print('The code was run!')