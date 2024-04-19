#pragma once
#include "NeuralNet_global.h"

/// USER_SECTION_START 1

/// USER_SECTION_END

// Debugging
#ifdef NDEBUG
	#define NN_CONSOLE(msg)
	#define NN_CONSOLE_FUNCTION(msg)
#else
	#include <iostream>

	#define NN_DEBUG
	#define NN_CONSOLE_STREAM std::cout

	#define NN_CONSOLE(msg) NN_CONSOLE_STREAM << msg;
	#define NN_CONSOLE_FUNCTION(msg) NN_CONSOLE_STREAM << __PRETTY_FUNCTION__ << " " << msg;
#endif

/// USER_SECTION_START 2

/// USER_SECTION_END

#ifdef NN_PROFILING
	#include "easy/profiler.h"
	#include <easy/arbitrary_value.h> // EASY_VALUE, EASY_ARRAY are defined here

	#define NN_PROFILING_BLOCK_C(text, color) EASY_BLOCK(text, color)
	#define NN_PROFILING_NONSCOPED_BLOCK_C(text, color) EASY_NONSCOPED_BLOCK(text, color)
	#define NN_PROFILING_END_BLOCK EASY_END_BLOCK
	#define NN_PROFILING_FUNCTION_C(color) EASY_FUNCTION(color)
	#define NN_PROFILING_BLOCK(text, colorStage) NN_PROFILING_BLOCK_C(text,profiler::colors::  colorStage)
	#define NN_PROFILING_NONSCOPED_BLOCK(text, colorStage) NN_PROFILING_NONSCOPED_BLOCK_C(text,profiler::colors::  colorStage)
	#define NN_PROFILING_FUNCTION(colorStage) NN_PROFILING_FUNCTION_C(profiler::colors:: colorStage)
	#define NN_PROFILING_THREAD(name) EASY_THREAD(name)

	#define NN_PROFILING_VALUE(name, value) EASY_VALUE(name, value)
	#define NN_PROFILING_TEXT(name, value) EASY_TEXT(name, value)

#else
	#define NN_PROFILING_BLOCK_C(text, color)
	#define NN_PROFILING_NONSCOPED_BLOCK_C(text, color)
	#define NN_PROFILING_END_BLOCK
	#define NN_PROFILING_FUNCTION_C(color)
	#define NN_PROFILING_BLOCK(text, colorStage)
	#define NN_PROFILING_NONSCOPED_BLOCK(text, colorStage)
	#define NN_PROFILING_FUNCTION(colorStage)
	#define NN_PROFILING_THREAD(name)

	#define NN_PROFILING_VALUE(name, value)
	#define NN_PROFILING_TEXT(name, value)
#endif

// Special expantion tecniques are required to combine the color name
#define CONCAT_SYMBOLS_IMPL(x, y) x##y
#define CONCAT_SYMBOLS(x, y) CONCAT_SYMBOLS_IMPL(x, y)



// Different color stages
#define NN_COLOR_STAGE_1 50
#define NN_COLOR_STAGE_2 100
#define NN_COLOR_STAGE_3 200
#define NN_COLOR_STAGE_4 300
#define NN_COLOR_STAGE_5 400
#define NN_COLOR_STAGE_6 500
#define NN_COLOR_STAGE_7 600
#define NN_COLOR_STAGE_8 700
#define NN_COLOR_STAGE_9 800
#define NN_COLOR_STAGE_10 900
#define NN_COLOR_STAGE_11 A100 
#define NN_COLOR_STAGE_12 A200 
#define NN_COLOR_STAGE_13 A400 
#define NN_COLOR_STAGE_14 A700 

namespace NeuralNet
{
	class NEURAL_NET_EXPORT Profiler
	{
	public:
		// Implementation defined in LibraryName_info.cpp to save files.
		static void start();
		static void stop();
		static void stop(const char* profilerOutputFile);
	};
}


// General
#define NN_GENERAL_PROFILING_COLORBASE Cyan
#define NN_GENERAL_PROFILING_BLOCK_C(text, color) NN_PROFILING_BLOCK_C(text, color)
#define NN_GENERAL_PROFILING_NONSCOPED_BLOCK_C(text, color) NN_PROFILING_NONSCOPED_BLOCK_C(text, color)
#define NN_GENERAL_PROFILING_END_BLOCK NN_PROFILING_END_BLOCK;
#define NN_GENERAL_PROFILING_FUNCTION_C(color) NN_PROFILING_FUNCTION_C(color)
#define NN_GENERAL_PROFILING_BLOCK(text, colorStage) NN_PROFILING_BLOCK(text, CONCAT_SYMBOLS(NN_GENERAL_PROFILING_COLORBASE, colorStage))
#define NN_GENERAL_PROFILING_NONSCOPED_BLOCK(text, colorStage) NN_PROFILING_NONSCOPED_BLOCK(text, CONCAT_SYMBOLS(NN_GENERAL_PROFILING_COLORBASE, colorStage))
#define NN_GENERAL_PROFILING_FUNCTION(colorStage) NN_PROFILING_FUNCTION(CONCAT_SYMBOLS(NN_GENERAL_PROFILING_COLORBASE, colorStage))
#define NN_GENERAL_PROFILING_VALUE(name, value) NN_PROFILING_VALUE(name, value)
#define NN_GENERAL_PROFILING_TEXT(name, value) NN_PROFILING_TEXT(name, value)


/// USER_SECTION_START 3

/// USER_SECTION_END