for(i=0; i<10;i++){
	newImage("Micrograph", "32-bit noise", 4096, 4096, 1);
	run("MRC writer", "save=./MicrographNoise.mrc");
	close();
}
