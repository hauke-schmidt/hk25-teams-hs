import pickle

class track_dates:
    def __init__(self,):
        self.dates={}

    def add_time(self,date,time):
        if date in self.dates.keys():
            if time not in self.dates[date]:
                self.dates[date].append(time)
        else:
            self.dates[date]=[time]
        

    def save_pickle(self, filename):
        f = open(filename, "wb")
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

def read_from_pickle(filename):
    f = open(filename, "rb")
    m = pickle.load(f)
    f.close()
    return m