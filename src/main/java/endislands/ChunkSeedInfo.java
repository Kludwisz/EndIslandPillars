package endislands;

import java.util.Scanner;

public class ChunkSeedInfo {
	public final long seed;
	public final int x;
	public final int z;
	public final SeedType type;
	
	public ChunkSeedInfo(long s, int x, int z, SeedType t) {
		this.seed = s;
		this.x = x;
		this.z = z;
		this.type = t;
	}
	
	public static ChunkSeedInfo read(Scanner fileInput, SeedType t) throws Exception {
		int x = fileInput.nextInt();
		int z = fileInput.nextInt();
		long seed = fileInput.nextLong();
		
		return new ChunkSeedInfo(seed, x, z, t);
	}
	
	public static enum SeedType {
		BOTTOM_SEED,
		TOP_SEED,
		NONE
	}
}
