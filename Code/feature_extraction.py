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
Local = False

if Local:
    PATH_MOVIES = '/Users/silviaromanato/Desktop/ServerMIPLAB/FilmFiles/'
else:
    PATH_MOVIES = '/media/miplab-nas2/Data2/Movies_Emo/FilmFiles/'



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
                #print('Read a new frame: ', success)
                count += 1
                if count % 500 == 0:
                    print(f"Frame {count} processed")
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

        if not os.path.exists(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/image_{movie_name}.png'):
            df_movie = FrameCapture(MOVIE_PATH)

            print(df_movie.head(30))

            # plot the average brightness and saturation and hue
            plt.figure()
            plt.plot(df_movie['average_brightness'], label='brightness')
            plt.plot(df_movie['average_saturation'], label='saturation')
            plt.plot(df_movie['average_hue'], label='hue')
            plt.legend()
            if Local:
                plt.savefig(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/image_{movie_name}.png')
            else:
                plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/image_{movie_name}.png')
            print('The plot was done and saved!')
        else:    
            print('The movie images were already analyzed!')

        # extract the audio from the movie
        print('Extracting the audio from the movie...')
        video = mp.VideoFileClip(MOVIE_PATH)
        video_duration = video.duration
        start_time = 1  # Start time in seconds
        end_time = video_duration  # End time, limited to video duration

        # Create the subclip within the specified time range
        clip = video.subclip(start_time, end_time)
        if Local:
            clip.audio.write_audiofile(f"/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Audio/audio_{movie_name}.wav")
            filename = f"/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Audio/audio_{movie_name}.wav"
        else:
            clip.audio.write_audiofile(f"/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio_{movie_name}.wav")
            filename = f"/media/miplab-nas2/Data2/Movies_Emo/Silvia/Audio/audio_{movie_name}.wav"
        print('The audio was extracted!')
        x, sr = librosa.load(filename, sr=22050)
        int(librosa.get_duration(x, sr) / 60)
        max_slice = 10
        window_length = max_slice * sr
        a = x[21 * window_length:22 * window_length]
        ipd.Audio(a, rate=sr)

        s_energy = np.array([sum(abs(x[i:i + window_length] ** 2)) for i in range(0, len(x), window_length)])
        print('The energy is: ', s_energy)

        plt.hist(s_energy)
        if Local:
            plt.savefig(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/energys_{movie_name}.png')
        else:
            plt.savefig(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/energys_{movie_name}.png')
        print('The plot of the energy was done and saved!')

        # add the energy to the dataframe
        df_movie['energy'] = s_energy

        # save the dataframe to a csv file
        if Local:
            df_movie.to_csv(f'/Users/silviaromanato/Desktop/SEMESTER_PROJECT/HigherOrder/Data/Output/movie_features_{movie_name}.csv', index=False)
        else:
            df_movie.to_csv(f'/media/miplab-nas2/Data2/Movies_Emo/Silvia/Data/Output/movie_features_{movie_name}.csv', index=False)

        print(df_movie.head(30))
        print('The code was run!')