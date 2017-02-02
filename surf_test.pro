TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

OPENCV_LIB_PATH = C:\opencv-build\install\x86\mingw\bin

INCLUDEPATH += C:\opencv-build\install\include


CONFIG(debug) {
    include(opencv_debug.pri)
}
CONFIG(release) {
    include(opencv_release.pri)
}
