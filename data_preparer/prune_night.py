import os, os.path

print("This program removes all images before 08:00 and after 17:00")


path_usr = input("Name of global path to files:")
if  path_usr == "":
    path_usr = "C:\\Users\\peterhs\\Documents\\fulehuk_images\\20190320"
    for i in range(20,27):
        
        path, dirs, files = next(os.walk(path_usr))

        for ffile in files:
            hour = int(ffile[11:13])
            if hour < 8:
                os.remove(os.path.join(path_usr, ffile))
            elif hour > 17:
                os.remove(os.path.join(path_usr, ffile))
            else:
                continue

        path_usr = path_usr[:-2] + "{:02d}".format(i)

else:
    path, dirs, files = next(os.walk(path_usr))

    for ffile in files:
        hour = int(ffile[11:13])
        if hour < 8:
            os.remove(os.path.join(path_usr, ffile))
        elif hour > 17:
            os.remove(os.path.join(path_usr, ffile))
        else:
            continue


print("All done!")
