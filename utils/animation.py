import os
import shutil
import subprocess
import io
from datetime import datetime
from PIL import Image
from tempfile import mkdtemp
import logging

from matplotlib.pyplot import Figure


def get_video_filename():
    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    timestring = datetime.today().strftime("%Y-%m-%d %Hh%Mm%S")
    return f'animation_{timestring}_{git_hash}'


def fig2img(fig):
    """Convert a matplotlib figure to a PIL Image and return it"""
    with io.BytesIO() as buf:
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        img.load()

    return img


class AnimatedImage(object):
    def __init__(self):
        super(AnimatedImage, self).__init__()
        self._frames = []

    def __len__(self):
        return len(self._frames)

    def add_frame(self, frame):
        if isinstance(frame, Figure):
            self.add_frame(fig2img(frame))
        else:
            self._frames.append(frame)

    def save(self, filename, fps, loop=False, comment=""):
        raise NotImplementedError()


class AnimatedGIF(AnimatedImage):
    def __init__(self):
        super(AnimatedGIF, self).__init__()

    def save(self, filename, fps, loop=False, comment=""):
        if not self._frames:
            raise Exception('No frames stored.')

        if not filename.lower().endswith('.gif'):
            filename += '.gif'

        self._frames[0].save(filename, save_all=True, append_images=self._frames[1:],
                             optimize=False, duration=1000 / fps, loop=loop, comment=comment)


class FFMPEGVideo(object):
    def __init__(self):
        super(FFMPEGVideo, self).__init__()
        self._ffmpeg = shutil.which('ffmpeg')
        self._workdir = mkdtemp(prefix='FFMPEGVideo.', dir=os.getcwd())
        self._framecounter = 0

        assert self._ffmpeg is not None  # should add ffmpeg\bin directory to PATH

    def __len__(self):
        return self._framecounter

    def add_frame(self, frame):
        filename = os.path.join(self._workdir, 'frame_{:08d}.png'.format(self._framecounter))

        if isinstance(frame, Figure):
            frame.savefig(filename, transparent=False)  # ffmpeg cannot deal properly with transparent pngs
        else:
            frame.save(filename)

        self._framecounter += 1

    def save(self, filename, fps, keep_frame_images=False):
        if self._framecounter == 0:
            raise Exception('No frames stored.')

        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'

        ffmpeg = subprocess.run([self._ffmpeg,
                                 '-y',  # force overwrite if output file exists
                                 '-framerate', '{}'.format(fps),
                                 '-i', os.path.join(self._workdir, 'frame_%08d.png'),
                                 '-c:v', 'libx264',
                                 '-preset', 'slow',
                                 '-crf', '17',
                                 '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p',
                                 filename,
                                 ])

        if ffmpeg.returncode != 0:
            logging.error('Running the following command failed with return code {}:\n\t{}'
                          .format(ffmpeg.returncode, ' '.join(ffmpeg.args)))
        elif not keep_frame_images:
            shutil.rmtree(self._workdir)


class ImageStack():
    def __init__(self, name: str, file_type: str = 'pdf'):
        self._framecounter = 0
        self._file_type = file_type
        self._workdir = os.path.join(os.getcwd(), name)

        if os.path.exists(self._workdir):
            shutil.rmtree(self._workdir)
        os.makedirs(self._workdir)

    def add_frame(self, frame):
        filename = os.path.join(self._workdir, 'frame_{:08d}.{}'.format(self._framecounter, self._file_type))

        if isinstance(frame, Figure):
            frame.savefig(filename, transparent=False)  # ffmpeg cannot deal properly with transparent pngs
        else:
            frame.save(filename)

        self._framecounter += 1
