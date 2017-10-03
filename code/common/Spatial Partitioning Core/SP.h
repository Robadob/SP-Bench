#ifndef SP_h
#define SP_h

extern "C" __declspec(dllexport)
void initSP();

extern "C" __declspec(dllexport)
void rebuildSP();

extern "C" __declspec(dllexport)
void insertSP();

extern "C" __declspec(dllexport)
void searchSP();

#endif //SP_h