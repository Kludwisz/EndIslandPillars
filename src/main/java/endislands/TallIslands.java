package endislands;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.seedfinding.latticg.util.LCG;
import com.seedfinding.mcbiome.biome.Biome;
import com.seedfinding.mcbiome.biome.Biomes;
import com.seedfinding.mcbiome.source.BiomeSource;
import com.seedfinding.mccore.rand.ChunkRand;
import com.seedfinding.mccore.state.Dimension;
import com.seedfinding.mccore.util.pos.BPos;
import com.seedfinding.mccore.util.pos.CPos;
import com.seedfinding.mccore.version.MCVersion;
import com.seedfinding.mcreversal.ChunkRandomReverser;

import endislands.ChunkSeedInfo.SeedType;
import endislands.MultiChunkHelper.Result;

public class TallIslands {
	/*
	 * Final results:
	 * 
	 * Phase 1 - 22-block high pillar
	 * 17760714566477  /execute in minecraft:the_end run tp @s 1040 80 16
	 * 
	 * Phase 2 - 26-block high pillar
	 * 33935436461067  /execute in minecraft:the_end run tp @s -2365344 80 -23592448
	 */

	private static final String root = "C:\\Users\\kludw\\eclipse-workspace\\endislands\\src\\main\\java\\endislands\\";
	
	public static void main(String [] args) throws Exception {
		// filterPopSeeds();
		// reverseAll();
		
		combineSeeds();
	}
	
	// ----------------------------------------
	// Phase 2 - multi-chunk end island pillar
	// ----------------------------------------
	
	private static void combineSeeds() throws Exception {
		ArrayList<ChunkSeedInfo> seedsBottom = new ArrayList<>();
		ArrayList<ChunkSeedInfo> seedsTop = new ArrayList<>();
		
		Scanner fin1 = new Scanner(new File(root + "popseedsBOT.txt"));
		while (fin1.hasNextLong()) {
			seedsBottom.add(ChunkSeedInfo.read(fin1, SeedType.BOTTOM_SEED));
		}
		fin1.close();
		
		Scanner fin2 = new Scanner(new File(root + "popseedsTOP.txt"));
		while (fin2.hasNextLong()) {
			seedsTop.add(ChunkSeedInfo.read(fin2, SeedType.TOP_SEED));
		}
		fin2.close();
		
		System.out.println("Seed files loaded!");
		
		// 2-chunk CRR
		for (ChunkSeedInfo seedInfoBottom : seedsBottom) {
			for (ChunkSeedInfo seedInfoTop : seedsTop) {
				// check if the chunk relative positions are good
				int posXor = seedInfoBottom.x ^ seedInfoTop.x ^ seedInfoBottom.z ^ seedInfoTop.z;
				if (posXor == 0) // xored either 0, 2 or 4 fifteens, all of which are bad
					continue;
				
				int dx = (seedInfoBottom.x - seedInfoTop.x) / 15;
				int dz = (seedInfoBottom.z - seedInfoTop.z) / 15;
				
				MultiChunkHelper twochunk = new MultiChunkHelper();
				List<Result> solutions = 
					twochunk.getWorldseedFromTwoChunkseeds(seedInfoBottom.seed, seedInfoTop.seed, dx*16, dz*16, null);
				
				if (solutions == null || solutions.isEmpty())
					continue;
				
				System.out.println(solutions.size() + " results:");
				for (Result res : solutions) {
					BiomeSource ebs = BiomeSource.of(Dimension.END, MCVersion.v1_16_1, res.getBitsOfSeed());
					Biome b = ebs.getBiomeForNoiseGen((res.getX()/16 << 2) + 2, 0, (res.getZ()/16 << 2) + 2);
					
					if (b.equals(Biomes.SMALL_END_ISLANDS)) {
						System.out.println(res.getBitsOfSeed() + "  /execute in minecraft:the_end run tp @s " + res.getX() + " 80 " + res.getZ());
						break;
					}
				}
				break;
			}
		}
	}
	
	// My original idea for phase 2 was to use the seeds found by the CUDA code
	// in phase 1 and just filter them down to get seeds where the end islands spawned
	// in the corner. This didn't work out.
	@Deprecated
	private static void filterCornerSeeds() throws Exception {
		Scanner fin = new Scanner(new File(root + "realpopseeds.txt"));
		ChunkRand rand = new ChunkRand();
		
		while (fin.hasNextLong()) {
			long popseed = fin.nextLong();
			// popseed ^= LCG.JAVA.multiplier;
			rand.setSeed(popseed);
			
			List<BPos> islands = getEndIslandPositions(rand, new CPos(0,0));
			int yOffset1 = placeIsland(rand);
			int yOffset2 = placeIsland(rand);
			
			int i1 = checkIsland(islands.get(0));
			int i2 = checkIsland(islands.get(1));
			
			if (i1 != 0) {
				System.out.println(popseed + "   variant " + i1 + "   positions: " + islands.get(0) + ",  " + islands.get(1));
			}
			if (i2 != 0) {
				System.out.println(popseed + "   variant " + i2 + "   positions: " + islands.get(0) + ",  " + islands.get(1));
			}
		}
		fin.close();
	}
	
	@Deprecated
	private static int checkIsland(BPos island) {
		if (island.getX() != 0 && island.getX() != 15) return 0;
		if (island.getZ() != 0 && island.getZ() != 15) return 0;
		if (island.getY() != 55 && island.getY() != 55+15) return 0;
		
		return island.getY() == 55 ? -1 : 1;
	}

	// -----------------------------------------
	// Phase 1 - single chunk end island pillar
	// -----------------------------------------
	
	private static void reverseAll() throws Exception {
		ChunkRand rand = new ChunkRand();
		
		Scanner fin = new Scanner(new File(root + "realpopseeds.txt"));
		while (fin.hasNextLong()) {
			long popseed = fin.nextLong();
			
			int cx = 65, cz = 0;
			while (true) {
				List<Long> structseeds = ChunkRandomReverser.reversePopulationSeed(popseed, cx << 4, cz << 4, MCVersion.v1_16_1);
				boolean found = false;
				for (long ss : structseeds) {
					BiomeSource ebs = BiomeSource.of(Dimension.END, MCVersion.v1_16_1, ss);
					Biome b = ebs.getBiomeForNoiseGen((cx << 2) + 2, 0, (cz << 2) + 2);
					if (b.equals(Biomes.SMALL_END_ISLANDS)) {
						System.out.println(ss + "  /execute in minecraft:the_end run tp @s " + cx*16 + " 80 " + cz*16);
						found = true;
						break;
					}
				}
				
				if (found) break;
				cz++;
			}
		}
	}

	// Phase 1 code
	private static void filterPopSeeds() throws Exception {
		ChunkRand rand = new ChunkRand();
		
		Scanner fin = new Scanner(new File(root + "popseedsPhase1.txt"));
		while (fin.hasNextLong()) {
			long popseed = fin.nextLong();
			popseed ^= LCG.JAVA.multiplier;
			
			// test if the popseed results in the generation of 2 connected islands
			rand.setSeed(popseed);
			
			// generate islands
			List<BPos> islands = getEndIslandPositions(rand, new CPos(0,0));
			int yOffset1 = placeIsland(rand);
			int yOffset2 = placeIsland(rand);
			int delta = 0;
			
			if (islands.get(0).getY() > islands.get(1).getY()) {
				delta = islands.get(0).getY() + yOffset1 - islands.get(1).getY();
			}
			else {
				delta = islands.get(1).getY() + yOffset2 - islands.get(0).getY();
			}
			
			if (delta == 0) {
				System.out.println(popseed);
			}
		}
		fin.close();
	}
	
	// ---------------------
	// Small End Island gen
	// ---------------------
	
	private static List<BPos> getEndIslandPositions(ChunkRand rand, CPos chunkPos) {
		ArrayList<BPos> positions = new ArrayList<>();
		BPos originalPos = chunkPos.toBlockPos(55);
		
		if (rand.nextInt(14) == 0) {
			positions.add(originalPos.add(rand.nextInt(16), rand.nextInt(16), rand.nextInt(16)));
			
			if (rand.nextInt(4) == 0) {
				positions.add(originalPos.add(rand.nextInt(16), rand.nextInt(16), rand.nextInt(16)));
			}
		}
		
		return positions;
	}
	
	
	private static int placeIsland(ChunkRand rand) {
		float r = (float)(rand.nextInt(3) + 4);
		int yOffset;

		for(yOffset = 0; r > 0.5F; --yOffset) {
			r = (float)((double)r - ((double)rand.nextInt(2) + 0.5D));
		}
		
		return yOffset;
	}
}
