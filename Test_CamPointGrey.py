from Lima import PointGrey
from Lima import Core

cam = PointGrey.Camera(13125072)
hwint = PointGrey.Interface(cam)
control = Core.control(hwint)

acq = control.acquisition()

# configure some hw parameters
hwint.setAutoGain(True)
hwint.se

# setting new file parameters and autosaving mode
saving=c.saving()

pars=saving.getParameters()
pars.directory='/buffer/lcb18012/opisg/test_lima'
pars.prefix='test1_'
pars.suffix='.edf'
pars.fileFormat=Core.CtSaving.EDF
pars.savingMode=Core.CtSaving.AutoFrame
saving.setParameters(pars)

# now ask for 10ms sec. exposure and 100 frames
acq.setAcqExpoTime(0.01)
acq.setNbImages(100)

acq.prepareAcq()
acq.startAcq()