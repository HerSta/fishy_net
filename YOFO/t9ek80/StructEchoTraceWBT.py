class StructEchoTraceWBT():
    def __init__(self, time):
        self.time = time
    def populate(self, data):
        self.Depth = data[0]
        self.TSComp = data[1] 
        self.TSUncomp = data[2]
        self.AlongshipAngle = data[3]
        self.AthwartshipAngle = data[4]
        self.sa = data[5]
        self.frequencyLimits = data[6:8]
        self.uncompensatedFrequencyResponse = data[8:1008]
        self.compensatedFrequencyResponse = data[1008:2008]
        self.withinMaxBeamCompensation = data[2008:3008]


