::Adjust ANN path
set ANN_PATH=E:\Computations\Reconstruction\HighRes\Data\Program\ann_bin
set PATH=%PATH%;%ANN_PATH%


call ..\VS2010\RMM\Release\RMM.exe -tps ..\Example FileCollection.txt TextureCoordinates.txt
pause
call ..\VS2010\RMM\Release\RMM.exe -rmm ..\Example FileCollection.txt LmksAll_Indices.txt CorrFileCollection.txt OuterBoundaryIndices.txt InnerBoundaryIndices.txt ..\Example\FrameworkRMM

pause