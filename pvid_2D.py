import ast
import os

from numpy.core.defchararray import endswith
import cv2
import imageio
from matplotlib import pyplot as pl
import numpy as np
#import pickle
import math
import scipy.ndimage
import toolz
from scipy.stats.stats import energy_distance, pearsonr
import copy
import matplotlib.animation as anim
from collections import deque, Counter
from scipy.stats import mode
import seaborn as sns
import pandas as pd
import itertools
import functools
from toolz.itertoolz import sliding_window
from matplotlib.collections import LineCollection
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import gaussian_filter



class Para:
    def __init__(self, timestmp, coord):
        self.location = [coord]
        self.lastcoord = coord
        self.timestamp = timestmp
        self.color = [np.random.random(), np.random.random(), np.random.random(
        )]
        self.completed = False
        self.waitindex = 0
        # waits at max 4 frames before giving up on particular para.
        self.waitmax = 10
        # max distance between para in consecutive frames in xy or xz vector space.
        self.thresh = 5
        self.double = False
#        self.area = area
        self.avg_velocity = 0

    def endrecord(self, clip):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULNDT FIND PARTNER
        if clip:
            self.location = self.location[:-self.waitmax]
        self.avg_velocity = np.mean(
            [np.linalg.norm(p) for p in np.diff(self.location, axis=0)])
        self.completed = True

    def nearby(self, contlist):
        #first, search the list of contours found in the image to see if any of them are near the previous position of this Para.
        if len(contlist) == 0:
            pcoords = np.array([])
        else:
            cont_arr = np.array(contlist)
            lastc = np.reshape(self.lastcoord, (1, 2))
            distance_past_thresh = np.where(
                np.sqrt(
                    np.sum(
                        (cont_arr-lastc)*(cont_arr-lastc), axis=1)) < self.thresh)
            pcoords = cont_arr[distance_past_thresh]
        
#if there's nothing found, add 1 to the waitindex, and say current position is the last position

        if pcoords.shape[0] == 0:
            self.location.append(self.lastcoord)
            if self.waitindex == self.waitmax:
                #end record if you've gone 'waitmax' frames without finding anything. this value greatly changes things. its a delicate balance between losing the para and waiting too long while another para enters
                self.endrecord(True)
            self.waitindex += 1

# this case is only one contour is within the threshold distance to this Para.
        elif pcoords.shape[0] == 1:
            newcoord = pcoords[0]
            index_to_pop = distance_past_thresh[0][0]
            self.location.append(newcoord)
            self.lastcoord = newcoord
            self.waitindex = 0
            contlist.pop(index_to_pop)

# this case is that two or more contours fit threshold distance. stop the record and mark it as a double.
        elif pcoords.shape[0] > 1:       
            self.endrecord(True)
            self.double = True
        return contlist


class ParaMaster2D():
    def __init__(self, directory, startframe, endframe):
       
        self.directory = directory
        self.all_xy_para = []
        self.xy_matrix = []
        self.distance_thresh = 100
        self.length_thresh = 50
        self.paravectors = []
        self.dots = []
        self.analyzed_frames = deque()
        self.analyzed_frames_raw = deque()
        self.af_mod = 5
        self.velocity_mags = []
        self.interp_indices = []
        self.sec_per_br_frame = 1
        self.frames_per_mode = 20
        self.backgrounds = []
        self.startover = False
        self.framewindow = [startframe, endframe]
        self.pvid_id = [file_id for file_id in
                        os.listdir(self.directory) if (file_id[-4:] == '.AVI' and file_id[0:4] != 'pvid')][0]
        self.total_numframes = 0
        self.frametimes = []
        self.framerate = np.loadtxt(self.directory + "/" +
                                     [file_id for file_id in
                                      os.listdir(self.directory) if file_id[-18:] ==
                                      'experiment_log.txt'][0]).astype(np.float)[1]
        self.ms_to_frame = 1000 / self.framerate
        self.US_times = (np.loadtxt(directory + "/" +
                                     [file_id for file_id in
                                      os.listdir(self.directory) if file_id[-12:] ==
                                      'US_times.txt'][0])).astype(np.int)
        self.CS_times = (np.loadtxt(directory + "/" + 
                                     [file_id for file_id in
                                      os.listdir(self.directory) if file_id[-12:] ==
                                      'CS_times.txt'][0])).astype(np.int)
        self.US_frames = []
        self.CS_frames = []
        
    def exporter(self):
        print('exporting ParaMaster')
        with open('paradata2D.pkl', 'wb') as file:
            pickle.dump(self, file)

# This function fits contours to each frame of the high contrast videos created in flparse2. Contours are filtered for proper paramecia size, then the locations of contours are compared to the current locations of paramecia. Locations of contours falling within a distance threshold of a known paramecia are appended to the location list for that para. At the end, individual paramecia records are established for each paramecia in the tank. Each para object is stored in the all_xy or all_xz para object list depending on plane.

    def get_frametimes(self):
        try:
            self.frametimes = list(np.load(self.directory+"/frametimes.npy"))
            self.US_frames = list(np.load(self.directory+"/US_frames.npy"))
            self.CS_frames = list(np.load(self.directory+"/CS_frames.npy"))
        except FileNotFoundError:
            framecounter = []
            self.frametimes = []
            pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
            while(True):
                ret, im = pvid_vcap.read()
                if not ret:
                    break
                img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                frame_id = img[0][0] + img[0][1] * 256 + img[0][2] * (256**2) + img[0][3] * (256**3)
                framecounter.append(frame_id)
            init_countval = framecounter[0]
            self.frametimes = [1000*(1/self.framerate) * (v-init_countval) for v in framecounter]
            np.save(self.directory+"/frametimes.npy", self.frametimes)
            self.US_frames = [np.abs(np.array(self.frametimes) - usf).argmin() for usf in self.US_times]
            self.CS_frames = [np.abs(np.array(self.frametimes) - csf).argmin() for csf in self.CS_times]
            np.save(self.directory+"/US_frames.npy", self.US_frames)
            np.save(self.directory+"/CS_frames.npy", self.CS_frames)
            pvid_vcap.release()


    def contrast_frame(self, frame_id, grayframe, prms):
    # obtains correct background window
        br = self.backgrounds[
            int(np.floor(frame_id / (self.framerate *
                                     self.sec_per_br_frame *
                                     self.frames_per_mode)))].astype(np.uint8)
        frame_avg = np.mean(grayframe)
        br_avg = np.mean(br)
        img_adj = (grayframe * (br_avg / frame_avg)).astype(np.uint8)
#        brsub = cv2.absdiff(img_adj, br)
        brsub = cv2.absdiff(grayframe.astype(np.uint8), br)
        cont_ = imcont(brsub, prms).astype(np.uint8)
        return cont_

    def make_backgrounds(self):
        print("IN BACKGROUNDS")
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        self.total_numframes = pvid_vcap.get(fc)
        ir_br_frames_to_mode = range(1,
                                     int(self.total_numframes),
                                     int(self.sec_per_br_frame * self.framerate))
        ir_temp = []
        for frame in ir_br_frames_to_mode:
            pvid_vcap.set(curr_frame, frame)
            ret, im = pvid_vcap.read()
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ir_temp.append(img)
            if len(ir_temp) % self.frames_per_mode == 0:
                self.backgrounds.append(
                    calc_mode(ir_temp,
                              np.zeros([1024, 1280])))
                ir_temp = []
        # accounts for leftovers at the end
        if ir_temp:
            self.backgrounds.append(calc_mode(ir_temp,
                                                np.zeros([1024, 1280])))
            ir_temp = []
        np.save(self.directory + '/backgrounds.npy', self.backgrounds)
        pvid_vcap.release()

    def findpara(self, params):
        print("FINDING PARA")
        # filtering_params give erosion, dilation, threshold, and area for para finding
        self.analyzed_frames = deque()
        completed_para_records = []
        filtering_params, area = params
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        print(self.pvid_id)
        self.total_numframes = pvid_vcap.get(fc)
        print("TOTAL NUMBER OF FRAMES")
        print(self.total_numframes)
        firstframe = True
        pvid_vcap.set(curr_frame, self.framewindow[0])
        for framenum in range(self.framewindow[0], self.framewindow[1]):
            if framenum % self.framerate == 0:
                print(framenum)
            ret, im = pvid_vcap.read()
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_contrasted = self.contrast_frame(framenum, im_gray,
                                                filtering_params)
            im_cont_color = cv2.cvtColor(im_contrasted, cv2.COLOR_GRAY2RGB)
            contours, hierarchy = cv2.findContours(
                im_contrasted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parafilter = [cv2.minEnclosingCircle(t)[0] for t in contours
                          if area[0] <= cv2.contourArea(t) < area[1]]
            for para in parafilter:
                cv2.circle(im_cont_color, (int(para[0]), int(para[1])), 2,
                           (255, 0, 255), -1)
            if framenum % self.af_mod == 0:
                self.analyzed_frames.append(im_cont_color)
            if firstframe:
                firstframe = False
                para_objects = [Para(framenum, pr) for pr in parafilter]
            else:
                xy_list = list(map(lambda n: n.nearby(parafilter), para_objects))
                newpara = [Para(framenum, pr) for pr in parafilter]
                para_objects = para_objects + newpara
                paramecia = list(filter(lambda x: x.completed, para_objects))
                completed_para_records = completed_para_records + paramecia
                para_objects = list(filter(lambda x: not x.completed, para_objects))
#current para list p_t and p_s are cleansed of records that are complete.
        self.analyzed_frames_raw = copy.deepcopy(self.analyzed_frames)
        self.create_coord_matrix()
        self.watch_event()
        pvid_vcap.release()
      #  modify = input('Modify Videos?: ')
        if not self.startover:
            modify = 's'
        else:
            modify = "n"
        if modify == 'r':
            modify = self.repeat_vids()
        if modify == 's':
            self.startover = True
            self.framewindow = [self.framewindow[0], int(self.total_numframes)]
            self.af_mod = 50
            pvid_vcap.release()
            return self.findpara(params)
        if modify == 'y':
            action = input('Enter new params: ')
            self.analyzed_frames = deque()
            pvid_vcap.release()
            return self.findpara(ast.literal_eval(action))
        if modify == 'x':
            return ""
        else:
            end_precs = list(map(lambda p: p.endrecord(False), para_objects))
            all_xy_para = completed_para_records + para_objects
            all_xy_para = sorted(all_xy_para, key=lambda x: len(x.location))
            all_xy_para.reverse()
            self.all_xy_para_raw = all_xy_para
            self.all_xy_para = [para for para in all_xy_para
                                if len(para.location) > self.length_thresh]
            print('Para Found in XY')
            print(len(self.all_xy_para))
            self.create_coord_matrix()
            self.label_para()

    def repeat_vids(self):
        self.watch_event()
        ra = input('Repeat, Yes, or No: ')
        if ra == 'r':
            return self.repeat_vids()
        elif ra == 'y':
            return 'y'
        else:
            return 'n'

    def refilter_para_records(self, reclen, avg_vel, xb, lane_b, xtravel):
        lane_ranges = [range(l[0], l[1]) for l in lane_b] 
        all_xy_records = copy.deepcopy(self.all_xy_para_raw)
        # this has to come first b/c p.location has to be nonzero
        all_xy_records = list(filter(lambda p: len(p.location) > reclen, all_xy_records))
        all_xy_records = list(filter(lambda p: p.avg_velocity > avg_vel, all_xy_records))
        all_xy_records = list(filter(
             lambda p: xb[0] < np.mean(np.array(p.location)[:, 0]) < xb[1], all_xy_records))
        all_xy_records = list(filter(
             lambda p: np.array(
                 [np.mean(
                     np.array(p.location)[:, 1]).astype(
                         np.int) in l for l in lane_ranges]).any(), all_xy_records))
        # want an x cumulative travel cutoff.
        all_xy_records = list(filter(lambda p: np.ptp(np.array(p.location)[:,0]) > xtravel, all_xy_records))
        self.all_xy_para = all_xy_records
        self.recalculate_mat_and_labels()
    
    def watch_event(self):
        self.analyzed_frames = copy.deepcopy(self.analyzed_frames_raw)
        self.label_para()
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        for fr, im in enumerate(self.analyzed_frames):
            cv2.imshow('vid', im)
            key_entered = cv2.waitKey(100)
            if key_entered == 50:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)


    def show_labeled_ir_frames(self, make_movie):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        pvid_vw = cv2.VideoWriter(self.directory + '/pvid_id_movie.AVI', fourcc, 30,
                                  (1280, 1024), True)
        cv2.namedWindow('vid', flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('vid', 20, 20)
        curr_frame = cv2.CAP_PROP_POS_FRAMES
        fc = cv2.CAP_PROP_FRAME_COUNT
        pvid_vcap = cv2.VideoCapture(self.directory + "/" + self.pvid_id)
        for frame in range(self.framewindow[0], self.framewindow[1]):
            ret, im = pvid_vcap.read()
            if frame % 10 == 0:
                for para_id, xyr in enumerate(self.xy_matrix[:, frame]):
                    # indicates a nan.
                    if math.isnan(xyr[0]):
                        pass
                    else:
                        cv2.putText(
                            im,
                            str(para_id),
                            (int(xyr[0]+7),
                             int(xyr[1])),
                            0,
                            .5,
                            color=(255, 0, 255))
                    if any(map(lambda x: x < frame < x + 200, self.US_frames)):
                        cv2.putText(
                            im,
                            "US ON",
                            (20, 40),
                            0,
                            1.5,
                            color=(0, 255, 255),
                            thickness=3)
                    if any(map(lambda x: x < frame < x + 200, self.CS_frames)):
                        cv2.putText(
                            im,
                            "CS ON",
                            (20, 40),
                            0,
                            1.5,
                            color=(0, 125, 125),
                            thickness=3)
                cv2.imshow('vid', im)
                if make_movie:
                    pvid_vw.write(im)
                key_entered = cv2.waitKey(1)
                if key_entered == 50:
                    break
                elif key_entered == 49:
                    cv2.waitKey(2000)
        pvid_vw.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        
    def label_para(self):
        temp_frames = deque()
        frame_num = 0
        while True:
            try:
                im = self.analyzed_frames.popleft()
            except IndexError:
                break
            if frame_num % self.framerate == 0:
                print(frame_num)
            para_id = 0
            # the row index of the para3Dcooords matrix defines xyz id.
            for para_id, xyr in enumerate(self.xy_matrix[:, frame_num]):
                # indicates a nan.
                if math.isnan(xyr[0]):
                    pass
                else:
                    cv2.putText(
                        im,
                        str(para_id),
                        (int(xyr[0]+5),
                         int(xyr[1])),
                        0,
                        .5,
                        color=(255, 255, 0))
            temp_frames.append(im)
            frame_num += self.af_mod
        self.analyzed_frames = temp_frames

    def recalculate_mat_and_labels(self):
        self.create_coord_matrix()


# have to account for fact that position may come before OR after

# sort the all_xy_para results by timestamp (lowest first)
# here, p

    def merge_records(self, rec1, rec2, relabel):
        p1 = self.all_xy_para[rec1]
        p2 = self.all_xy_para[rec2]
        # this is correct
        time_win1 = p1.timestamp + len(p1.location)
        if time_win1-1 == p2.timestamp:
            p2loc = p2.location[1:]
        else:
            p2loc = p2.location
        p1.location = p1.location + [np.array([np.nan, np.nan]) for i in range(
            time_win1, p2.timestamp)] + p2loc
        del self.all_xy_para[rec2]
        if relabel:
            self.recalculate_mat_and_labels()
            
    def manual_remove(self, reclist, relabel):
        reclist.sort()
        reclist.reverse()
        for rec in reclist:
            del self.all_xy_para[rec]
        if relabel:
            self.recalculate_mat_and_labels()

    def create_coord_matrix(self):
        xy_matrix = np.zeros([len(self.all_xy_para),
                              int(self.total_numframes)],
                             dtype='2f')
        for row, rec in enumerate(self.all_xy_para):
            # this is correct
            xy_coords = [(np.nan, np.nan) for i in range(int(self.total_numframes))]
            xy_coords[rec.timestamp:rec.timestamp+len(
                rec.location)] = rec.location
            xy_matrix[row] = xy_coords
        self.xy_matrix = xy_matrix
        np.save(self.directory + '/para_matrix.npy', xy_matrix)
        

def xymat_to_dataframe(xy_matrix, drct):
    df_dict = {'frames': list(itertools.chain.from_iterable([
        range(xy_matrix.shape[1]) for i in range(xy_matrix.shape[0])])),
               'xcoords': list(itertools.chain.from_iterable([x for x in xy_matrix[:,:, 0]])),
               'ycoords': list(itertools.chain.from_iterable([y for y in xy_matrix[:,:, 1]])),
               'para_id': list(itertools.chain.from_iterable([i*np.ones(xy_matrix.shape[1]) for i in range(xy_matrix.shape[0])]))}
    df = pd.DataFrame(df_dict)
    df.to_csv(drct+"para_coords.csv")
    return df


def plot_x_tsplot(pmaster):
    # need random color from para object and the xy_matrix
    # dataframe needs to be like ParaID, frame, xcoord, ycoord
    fig, ax = pl.subplots(1, 2)
    df = xymat_to_dataframe(pmaster.xy_matrix)
    fig, ax = pl.subplots()
    sns.lineplot(data=df, x='frames', y='xcoords', ax=ax)
    for usframe, csframe in zip(pmaster.US_frames, pmaster.CS_frames):
        ax.vlines(x=usframe, ymin=0, ymax=1024, color='yellow')
        ax.vlines(x=csframe, ymin=0, ymax=1024, color='gray')
    pl.show()
    return df

def plot_all_recs(directory):
    framerate = np.loadtxt(
        directory + "/" + [file_id for file_id in os.listdir(
            directory) if file_id[-18:] == 'experiment_log.txt'][0]).astype(np.float)[1]
    ms_to_frame = 1000 / framerate
    fig, ax = pl.subplots()
    frametimes = np.load(directory +"/frametimes.npy")
    pmat = np.load(directory+"/para_matrix.npy")
    us_frames = np.load(directory + "/US_frames.npy")
    cs_frames = np.load(directory + "/CS_frames.npy")
    for p_id, xy_rec in enumerate(pmat):
        para = xy_rec
        xcoords = xy_rec[:,0]
        ax.plot(xcoords)
        last_nonnan = len(xcoords) - np.argmax([not math.isnan(x) for x in xcoords[::-1]]) - 1
        ax.text(last_nonnan, xcoords[last_nonnan], str(p_id))
    for usframe, csframe in zip(us_frames, cs_frames):
        ax.vlines(x=usframe, ymin=0, ymax=2000, color='yellow')
        ax.vlines(x=csframe, ymin=0, ymax=2000, color='gray')
    pl.savefig(directory+"/allrecsplot.png")
    pl.show()
    


    
def plot_xy_vs_t(pmat, rec):
    fig, ax = pl.subplots()
    xy_timeseries = pmat[rec]
    ts = range(len(xy_timeseries))
    x = np.array([x[0] for x in xy_timeseries]) # if not math.isnan(x[0])])
    y = np.array([y[1] for y in xy_timeseries]) # if not math.isnan(y[1])])
    segs = make_segments(x, y)
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))
    cmap = pl.get_cmap('viridis')
    norm = pl.Normalize(0.0, 1.0)
    linewidth = 3
    alpha = 1.0
    lc = LineCollection(segs,
                        array=np.linspace(0.0, 1.0, len(x)),
                        cmap=cmap,
                        norm=norm,
                        linewidth=linewidth,
                        alpha=alpha)
    ax.add_collection(lc)
    pl.show()
    return segs

    

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Interface to LineCollection:
    


def aggregate_us_cs_responses(directories, plot_or_not):
  #  directories = 00
    framerate = np.loadtxt(
        directories[0] + "/" + [file_id for file_id in os.listdir(
            directories[0]) if file_id[-18:] == 'experiment_log.txt'][0]).astype(np.float)[1]
    ms_to_frame = 1000 / framerate
    ind_cs_frames = (np.loadtxt(directories[0] + "/" + 
                                [file_id for file_id in
                                 os.listdir(directories[0]) if file_id[-12:] ==
                                 'CS_times.txt'][0]) / 10).astype(np.int)
    fig, ax = pl.subplots(2, len(ind_cs_frames))
    baseline = 200
    bounds = 5
    vel_CS = {i: [] for i, j in enumerate(ind_cs_frames)}
    vel_US = {i: [] for i, j in enumerate(ind_cs_frames)}
    vel_CS_w_id = {i: [] for i, j in enumerate(ind_cs_frames)}
    p_id_counter = 0
    for d_ind, directory in enumerate(directories):
        xy_matrix = np.load(directory + "/para_matrix.npy")
        us_frames = np.load(directory + "/US_frames.npy")
        cs_frames = np.load(directory + "/CS_frames.npy")
        for stim_ind, (csf, usf) in enumerate(zip(cs_frames, us_frames)):
            for (p_id, xyr) in enumerate(xy_matrix):
                kern = Gaussian1DKernel(3)
              #  x_vel = convolve(np.diff(xyr, axis=0)[:,0], kern)
              #  y_vel = convolve(np.diff(xyr, axis=0)[:,1], kern)
              #  cs_vdots = [np.dot(v1, v2) for v1, v2 in sliding_window(2, cs_vel_vectors)]
              #  vel_CS[c_ind].append(cs_vdots)
                xv_cs = np.diff(xyr[csf-baseline:csf+200], axis=0)[:,0]
                xv_us = np.diff(xyr[usf-baseline:usf+200], axis=0)[:,0]
                if np.nanmean(xv_cs[0:baseline]) < 0:
                    xv_cs *= -1
                # filters for broken records but not as harshly b/c its a longer window
                if sum(np.array([math.isnan(xval) for xval in xv_cs])) < 300 and not xv_cs.size == 0:
              #  if not np.array([math.isnan(xval) for xval in xv_cs]).any() and not xv_cs.size == 0:
                    print(p_id + p_id_counter)
                    print(sum(np.array([math.isnan(xval) for xval in xv_cs])))
                    vel_CS[stim_ind].append(convolve(xv_cs, kern, boundary='extend'))
                    vel_CS_w_id[stim_ind].append((p_id+p_id_counter, convolve(xv_cs, kern, boundary='extend')))
                if not np.array([math.isnan(xval) for xval in xv_us]).any() and not xv_us.size == 0:
                    vel_US[stim_ind].append(convolve(xv_us, kern, boundary='extend'))                
              #  us_vdots = [np.dot(v1, v2) for v1, v2 in sliding_window(2, us_vel_vectors)]
              #  vel_US[u_ind].append(us_vdots)
        p_id_counter += xy_matrix.shape[0]
    if plot_or_not:
        for ind, cs in enumerate(ind_cs_frames):
            ts_plot(vel_CS[ind], ax[0, ind], 'frame', 'x')
            ax[0, ind].vlines(x=baseline, ymin=-.5, ymax=.5, color='gray')
            ax[0, ind].set_ylim(-.5, .5)
            ts_plot(vel_US[ind], ax[1, ind], 'frame', 'x')
            ax[1, ind].set_ylim(-.5, .5)
            ax[1, ind].vlines(x=baseline, ymin=-.5, ymax=.5, color='yellow')
        pl.savefig("E:/ParaBehaviorData/aggregated_responses.png")
        pl.show()
    us_cs_dict_to_csv(vel_CS_w_id, cs_frames)
    return vel_CS_w_id, cs_frames

def us_cs_dict_to_csv(p_dict, cs_frames):
    directory = "E:/ParaBehaviorData/"
    window_size = len(p_dict[0][0][1])
    frame_windows = []
    para_ids = []
    para_velocity = []
    trial_id = []
    for cs_ind, csf in enumerate(cs_frames):
        for rec in p_dict[cs_ind]:
           # frame_windows.append(range(csf-200, csf-200+window_size))
            frame_windows.append(range(-200, -200+window_size))
            para_ids.append(list(rec[0]*np.ones(window_size)))
            para_velocity.append(rec[1])
            trial_id.append(list(cs_ind*np.ones(window_size)))
    # this is just compiling lists into a single list
    df_dict = {'frame_index': list(itertools.chain.from_iterable(frame_windows)),
               'para_id': list(itertools.chain.from_iterable(para_ids)),
               'trial': list(itertools.chain.from_iterable(trial_id)),
               'velocity': list(itertools.chain.from_iterable(para_velocity))}
    df = pd.DataFrame(df_dict)
    df.to_csv(directory+"para_aggregate.csv")


def ts_plot(list_of_lists, ax, lab_x, lab_y):
    index_list = list(
        itertools.chain.from_iterable(
            [range(len(arr)) for arr in list_of_lists]))
    id_list = [(ind*np.ones(len(arr))).tolist() for ind, arr in enumerate(list_of_lists)]
    ids_concatenated = list(itertools.chain.from_iterable(id_list))
    #this works if passed np.arrays instead of lists
    value_list = list(itertools.chain.from_iterable(list_of_lists))
    df_dict = {lab_x: index_list, 
               lab_y: value_list}
    df = pd.DataFrame(df_dict)
    sns.lineplot(data=df, x=lab_x, y=lab_y, ax=ax)
    return df


def imcont(image, params):
    thresh, erode_win, dilate_win, med_win = params
    r, th = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    ek = np.ones([erode_win, erode_win]).astype(np.uint8)
    dk = np.ones([dilate_win, dilate_win]).astype(np.uint8)
    er = cv2.erode(th, ek)
    dl = cv2.dilate(er, dk)
    md = cv2.medianBlur(dl, med_win)
    return md


def calc_median(deq, nump_arr):
    # so k are the values, j are the indicies.
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j] = np.median([x[j] for x in deq], axis=0)
    return nump_arr


def calc_mode(deq, nump_arr):
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j, :] = mode(np.array([x[j, :] for x in deq]))[0]
    return nump_arr


def same_lane(ppos1, ppos2, lane_borders):
    p1_lane = [lane for lane, lbs in enumerate(
        lane_borders) if (lbs[0] - 10 <= ppos1[1] <= lbs[1] + 10)]
    p2_lane = [lane for lane, lbs in enumerate(
        lane_borders) if (lbs[0] - 10 <= ppos2[1] <= lbs[1] + 10)]
    if p1_lane and p2_lane:
        if p1_lane == p2_lane:
            return True
        else:
            return False
    else:
        return False

def re_automerge(pmaster, x_boundaries, lane_boundaries):
    pmaster.refilter_para_records(100, .25, x_boundaries, lane_boundaries, 50)
    automerge_records(pmaster, lane_boundaries)
    pmaster.show_labeled_ir_frames(True) 
    plot_all_recs(pmaster.directory)

def automerge_records(pmaster, lane_borders):
    spacethresh = 1000
    min_dist = 20
    for ind, para in enumerate(pmaster.all_xy_para):
        base_ts = para.timestamp+len(para.location)-1
        base_pos = para.location[-1]
        time_between_recs_forward = list(map(lambda cpara: cpara.timestamp-base_ts if (
            same_lane(base_pos, cpara.location[0], lane_borders) and cpara.timestamp-base_ts >= 0 and (base_pos != cpara.location[-1]).all()) else np.inf, pmaster.all_xy_para))
        try:
            closest_in_time_forward = np.where(scipy.stats.rankdata(time_between_recs_forward) < 2)[0]
            if len(closest_in_time_forward) > 1:
                pos_cand1 = pmaster.all_xy_para[closest_in_time_forward[0]].location[0]
                pos_cand2 = pmaster.all_xy_para[closest_in_time_forward[1]].location[0] 
                dist_from_curr_para = [np.linalg.norm(pos_cand1-base_pos), np.linalg.norm(pos_cand2-base_pos)]
                if dist_from_curr_para[0] > dist_from_curr_para[1]:
                    closest_in_time_forward = closest_in_time_forward[1]
                else:
                    closest_in_time_forward = closest_in_time_forward[0]
            else:
                closest_in_time_forward = closest_in_time_forward[0]
            # if they are the same, give back the closer of the two instead of the first. 
            # do the same with backward!
        

        except IndexError:
            continue
        print(closest_in_time_forward)
        closest_t_para = pmaster.all_xy_para[closest_in_time_forward]
        time_between_recs_backward = list(map(lambda cpara: closest_t_para.timestamp-(cpara.timestamp+len(cpara.location)-1) if (
            same_lane(closest_t_para.location[0], cpara.location[-1], lane_borders) and closest_t_para.timestamp-(
                cpara.timestamp+len(cpara.location)-1) >= 0 and (closest_t_para.location[-1] != cpara.location[-1]).all()) else np.inf, pmaster.all_xy_para))
        closest_in_time_backward = np.where(scipy.stats.rankdata(time_between_recs_backward) < 2)[0]
         
        if ind in closest_in_time_backward:
          #  if len(closest_in_time_backward) > 1:
           #    p1_loc = pmaster.all_xy_para[closest_in_time_backward[0]].location[-1] 

            print("MERGING" + str(closest_in_time_forward))
            pmaster.merge_records(ind, closest_in_time_forward, False)
            return automerge_records(pmaster, lane_borders)
        else:
            continue

        # near_and_prox = list(map(lambda cp: (closest_in_time[cp[0]] == 1) and
        #     (np.linalg.norm(cp[1].location[0] - base_pos) < min_dist + spacethresh*(1-math.exp(-.001*(cp[1].timestamp-base_ts)))), list(enumerate(pmaster.all_xy_para))))
        # try:
        #     cand_match = near_and_prox.index(True)
        #     if cand_match == ind:
        #         cand_match = near_and_prox[ind+1:].index(True)
        # except ValueError:
        #     continue
       
    print("automerge complete")
    pmaster.recalculate_mat_and_labels()

def grab_random_frames(directory, num_random_frames):
    pvid_id = [file_id for file_id in
                         os.listdir(directory) if (file_id[-4:] == '.AVI' and file_id[0:4] != 'pvid')][0]
    curr_frame = cv2.CAP_PROP_POS_FRAMES
    fc = cv2.CAP_PROP_FRAME_COUNT
    pvid_vcap = cv2.VideoCapture(directory + "/" + pvid_id)
    total_numframes = pvid_vcap.get(fc)
    random_frames = [np.random.randint(total_numframes) for i in range(num_random_frames)]
    frameholder = []
    for frame in random_frames:
        pvid_vcap.set(curr_frame, frame)
        ret, im = pvid_vcap.read()
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        frameholder.append(img)
    md = calc_median(frameholder, np.zeros([1024, 1280]))
    pvid_vcap.release()
    return md

def find_lanes_in_br(br, delta_thresh, blurval):
#    br = cv2.GaussianBlur(np.load(drct + "/backgrounds.npy").astype(np.uint8)[3], (5,5), sigmaX=1, sigmaY=1)
    if delta_thresh < 5:
        return [], []
    #br = np.load(drct + "/backgrounds.npy").astype(np.uint8)[3]
    lines_through_lanes = []
    fig, ax = pl.subplots()
    for c in range(400, br.shape[1]-400, 20):
        filtered_lane_vals = gaussian_filter(br[:, c], 3)
        lines_through_lanes.append(filtered_lane_vals)
    light_profile = np.median(lines_through_lanes, axis=0)
    ax.plot(light_profile)
    delta_light_profile = [ind for ind, lp in enumerate(
        sliding_window(10, light_profile)) if np.abs(lp[0]-lp[9]) > delta_thresh]
    edges = [ind[0] for ind in sliding_window(2, delta_light_profile) if ind[0] - ind[1] < -50] + [delta_light_profile[-1]]
    if edges[0] > 100:
        edges = [0] + edges
    if edges[-1] < 924:
        edges = edges + [1024]
    print(edges)
    if len(edges) < 10:
        return find_lanes_in_br(br, delta_thresh-1, blurval+.5)
    if np.median(light_profile[edges[0]:edges[1]]) > np.median(light_profile[edges[1]:edges[2]]):
        lane_borders = [(edges[i], edges[i+1]) for i in range(0, len(edges) -1, 2)]
    else:
        lane_borders = [(edges[i], edges[i+1]) for i in range(1, len(edges) -1, 2)]
    lane_borders = [[bounds[0]-15, bounds[1]+15] for bounds in lane_borders]
    middle_lanes = lane_borders[1:4]
    x_profile = gaussian_filter(np.median([br[np.mean(lb).astype(np.int), :] for lb in middle_lanes], axis=0), 3)
    delta_x_profile = [ind for ind, xp in enumerate(
        sliding_window(10, x_profile)) if np.abs(xp[0]-xp[9]) > 10]
    x_edges = [ind[0] for ind in sliding_window(2, delta_x_profile) if (ind[0]-ind[1] < -50) and ((ind[0] < 250) or (ind[0] > 950))] + [delta_x_profile[-1]]
    print(x_edges)
    x_leftbound = [ind for ind in x_edges if ind < 250]
    x_rightbound = [ind for ind in x_edges if ind > 950]
    if not x_leftbound and not x_rightbound:
        return lane_borders, []
    elif not x_leftbound:
        xr = np.argmax([x_profile[i] for i in x_rightbound])
        x_right = x_rightbound[xr]
        x_left = x_right - 1050
    elif not x_rightbound:
        xl = np.argmax([x_profile[i] for i in x_leftbound])
        x_left = x_leftbound[xl]
        x_right = x_left + 1050
    else:
        xl = np.argmax([x_profile[i] for i in x_leftbound])
        xr = np.argmax([x_profile[i] for i in x_rightbound])
        x_left = x_leftbound[xl]
        x_right = x_rightbound[xr]
   # ax.scatter([x for x in edges], np.zeros(len(edges)), color='r')
   # ax.plot(x_profile, color='g')
   # ax.scatter(x_edges, np.zeros(len(x_edges)), color='k')
   # pl.show()
    return lane_borders, [x_left, x_right]
    


def make_paramaster(directory,
                    start_ind,
                    end_ind):
    paramaster = ParaMaster2D(directory, start_ind, end_ind)
    paramaster.get_frametimes()
    try:
        paramaster.backgrounds = np.load(directory+"/backgrounds.npy")
    except:
        paramaster.make_backgrounds()
    paramaster.findpara([[3, 3, 3, 5], [5, 100]])
    return paramaster



if __name__ == '__main__':

    # GOOD STARTING PARAMS
    dir_input = input("Enter Directories:  ")
    directory_list = ["E:/ParaBehaviorData/" + di for di in eval(dir_input)]
    lane_boundary_list = []
    x_boundary_list = []
    for directory in directory_list:
        try:
            lane_boundaries = list(np.load(directory + "/lane_boundaries.npy"))
            x_boundaries = list(np.load(directory + "/x_boundaries.npy"))
        except FileNotFoundError:
            br = grab_random_frames(directory, 5)
            try:
                lane_boundaries, x_boundaries = find_lanes_in_br(br, 10, 3)
            except IndexError:
                lane_boundaries = []
                x_boundaries = []
        if len(lane_boundaries) < 5:
            lane_boundaries = eval(input("Enter Lane Boundaries For " + directory + ":"))
        if len(x_boundaries) < 2:
            x_boundaries = eval(input("Enter X Boundaries For " + directory + ":"))     
        np.save(directory + "/lane_boundaries.npy", lane_boundaries)
        np.save(directory + "/x_boundaries.npy", x_boundaries)
        lane_boundary_list.append(lane_boundaries)
        x_boundary_list.append(x_boundaries)
    for d_ind, directory in enumerate(directory_list):
        lane_boundaries = lane_boundary_list[d_ind]
        x_boundaries = x_boundary_list[d_ind]
        print(directory)
        pmaster = make_paramaster(directory,
                                  1,
                                  50)
        
        pmaster.refilter_para_records(100, .25, x_boundaries, lane_boundaries, 50)
        automerge_records(pmaster, lane_boundaries)
        pmaster.show_labeled_ir_frames(True) 
  #  for d in directory_list:
  #      plot_all_recs(d) 
    aggregate_us_cs_responses(directory_list, False)
#drclist = ['E:/ParaBehaviorData/test70_US_' + d for d in ['a', 'b', 'c', 'd', 'f']]
#aggregate_us_cs_responses(drclist, False)
    
