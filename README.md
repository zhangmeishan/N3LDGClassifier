N3LDGClassifier
===========================
A simple guide for EGN3LDG: text classification 
* Simple CNN and LSTM classifiers are implemented.
* Checkgrad in cpp file can be commented out as a pratical classifier.
* add_definitions( -DUSE_FLOAT ) must be commented out for Checkgrad.
* If we add new neural modules in EGN3LDG, should check the module gradients by the clasifier.
 
## Run:
(1) cd build;    
(2) rm -rf *;    
(3) cmake ..  (if MKL is installed, check "[set(MKL_ROOT /opt/intel/mkl)](CMakeLists.txt)", and run cmake .. -DMKL=True);    
(4) make;    


If you have any problem, please send an email to mason.zms@gmail.com.
