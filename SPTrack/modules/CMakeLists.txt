ADD_LIBRARY(sp_player sp_player.cpp)
ADD_LIBRARY(sp_dct sp_dct.cpp)
ADD_LIBRARY(sp_hmm sp_hmm.cpp)


INCLUDE_DIRECTORIES(${SPTrack_SOURCE_DIR}/include)
LINK_DIRECTORIES(${SPTrack_SOURCE_DIR}/modules)
