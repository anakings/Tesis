import os
from csv import writer

class logs():
	def createFolder(self,newpath):
		if not os.path.exists(newpath):
			os.makedirs(newpath)

	def cleanFile(self,fileName):
		open(fileName, 'w').close()

	def writeFile(self, fileName, List):
		with open(fileName, 'a', newline='') as f_object: 
			# Pass this file object to csv.writer() 
			# and get a writer object 
			writer_object = writer(f_object) 
			# Pass the list as an argument into 
			# the writerow() 
			writer_object.writerow(List) 
			#Close the file object 
			f_object.close()

	def readFile(self,fileName):
		with open(fileName, 'rt') as f:
				data = reader(f)
				for row in data:
					print(row)
	
	 
